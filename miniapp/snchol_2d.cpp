#include <fstream>
#include <array>
#include <random>
#include <fstream>
#include <iostream>
#include <set>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <exception>
#include <map>
#include <mutex>
#include <tuple>
#include <memory>
#include <utility>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <scotch.h>
#include <gtest/gtest.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "runtime.hpp"
#include "util.hpp"
#include "mmio.hpp"
#include "communications.hpp"
#include "runtime.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef SparseMatrix<double> SpMat;
typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 0;
int LOG = 0;
string FILENAME = "neglapl_2_128.mm";
int N_LEVELS = 10;
int N_THREADS = 4;
int BLOCK_SIZE = 10;
int PROWS = 1;
int PCOLS = 1;
string FOLDER = "./profiles";

struct range
{
    int lb;
    int ub;
    int k;
};

int3 lower(int3 kij)
{
    int k = kij[0];
    int i = kij[1];
    int j = kij[2];
    if (i >= j)
        return kij;
    else
        return {k, j, i};
};

// Find the positions of c_rows into p_rows
// This could be faster since both are sorted
vector<int> get_subids(vector<int> &c_rows, vector<int> &p_rows)
{
    int cn = c_rows.size();
    vector<int> subids(cn);
    int l = 0;
    for (int i = 0; i < cn; i++)
    {
        while (c_rows[i] != p_rows[l])
        {
            l++;
        }
        assert(l < p_rows.size());
        assert(p_rows[l] == c_rows[i]);
        subids[i] = l;
    }
    return subids;
};

VectorXd random(int size, int seed)
{
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    VectorXd x(size);
    for (int i = 0; i < size; i++)
    {
        x[i] = dist(rng);
    }
    return x;
}

struct Bloc
{
    // Bloc data
    unique_ptr<MatrixXd> matA; // Lower triangular part only if on diagonal
    vector<int> rows;          // Global rows
    vector<int> cols;          // Global cols 
    int n_accumulate;          // How many blocs to accumulate into this guy 
    
    // Accumulation data    
    mutex to_accumulate_mtx;   // Mutex to guard to_accumulate below
                               // The blocs to be accumulated on this Bloc
    map<int, unique_ptr<MatrixXd>> to_accumulate; 
    // Accumulation debugging
    atomic<bool> accumulating_busy;
    atomic<int> accumulated;
    
    Bloc() : matA(nullptr), n_accumulate(0), accumulating_busy(false), accumulated(0){};
    MatrixXd *A() { return matA.get(); }

    // Allocate and fill
    void allocate() {
        matA = make_unique<MatrixXd>(rows.size(), cols.size());
        matA->setZero();
    }
};

struct Node
{
    // Node data
    int start;            // Global row/col
    int size;             // Global row/col
    int end;              // Global row/col
    vector<int> nbrs;     // Global node id
    vector<int> children; // Global node id
    int parent;           // Global node id
    VectorXd xsol;        // Solution on this node
    Node(int s_, int l_) : start(s_), size(l_) {
        end = start + size;
    };
};

struct DistMat
{
    map<int,  unique_ptr<Node>> nodes; // nodes[i]   = ith node (~= diagonal bloc)
    map<int2, unique_ptr<Bloc>> blocs; // blocs[i,j] = non zero part of the matrix

    VectorXi perm;
    SpMat A;
    SpMat App;

    int nblk;
    int prows;
    int pcols;

    // 2D processor mapping
    int ij2rank(int2 ij)
    {
        int i = ij[0];
        int j = ij[1];
        assert(i < nblk && i >= 0);
        assert(j < nblk && j >= 0);
        return i % prows + prows * (j % pcols);
    }

    // How many accumulations to do at every bloc
    int n_to_accumulate(int2 ij)
    {
        return blocs.at(ij)->n_accumulate;
    }
    int accumulated(int2 ij)
    {
        return blocs.at(ij)->accumulated.load();
    }

    // Some statistics/timings
    atomic<long long> gemm_us;
    atomic<long long> trsm_us;
    atomic<long long> potf_us;
    atomic<long long> scat_us;
    atomic<long long> allo_us;

    DistMat(string filename, int nlevels, int block_size, int _prows, int _pcols) : nblk(0), prows(_prows), pcols(_pcols), gemm_us(0), trsm_us(0), potf_us(0), scat_us(0), allo_us(0)
    {
        cout << "Matrix file " << filename << endl;
        // Read matrix
        A = mmio::sp_mmread<double, int>(filename);
        // Initialize & prepare
        int N = A.rows();
        int nnz = A.nonZeros();
        // Create rowval and colptr
        VectorXi rowval(nnz);
        VectorXi colptr(N + 1);
        int k = 0;
        colptr[0] = 0;
        for (int j = 0; j < N; j++)
        {
            for (SpMat::InnerIterator it(A, j); it; ++it)
            {
                int i = it.row();
                if (i != j)
                {
                    rowval[k] = i;
                    k++;
                }
            }
            colptr[j + 1] = k;
        }
        // Create SCOTCH graph
        SCOTCH_Graph *graph = SCOTCH_graphAlloc();
        int err = SCOTCH_graphInit(graph);
        assert(err == 0);
        err = SCOTCH_graphBuild(graph, 0, N, colptr.data(), nullptr, nullptr, nullptr, k, rowval.data(), nullptr);
        assert(err == 0);
        err = SCOTCH_graphCheck(graph);
        assert(err == 0);
        // Create strat
        SCOTCH_Strat *strat = SCOTCH_stratAlloc();
        err = SCOTCH_stratInit(strat);
        assert(err == 0);
        assert(nlevels > 0);
        string orderingstr = "n{sep=(/levl<" + to_string(nlevels - 1) + "?g:z;),ose=b{cmin=" + to_string(block_size) + "}}";
        cout << "Using ordering " << orderingstr << endl;
        // string orderingstr = "n{sep=(/levl<" + to_string(nlevels-1) + "?g:z;)}";
        err = SCOTCH_stratGraphOrder(strat, orderingstr.c_str());
        assert(err == 0);
        // Order with SCOTCH
        VectorXi permtab(N);
        VectorXi peritab(N);
        VectorXi rangtab(N + 1);
        VectorXi treetab(N);
        err = SCOTCH_graphOrder(graph, strat, permtab.data(), peritab.data(), &nblk, rangtab.data(), treetab.data());
        assert(err == 0);
        assert(nblk >= 0);
        treetab.conservativeResize(nblk);
        rangtab.conservativeResize(nblk + 1);
        // Permute matrix
        App = permtab.asPermutation() * A * permtab.asPermutation().transpose();
        perm = permtab;
        VectorXi i2irow(N);
        double mean = 0.0;
        int mini = App.rows();
        int maxi = -1;
        // Create all nodes
        for (int i = 0; i < nblk; i++)
        {
            nodes[i] = make_unique<Node>(rangtab[i], rangtab[i + 1] - rangtab[i]);
            for (int j = rangtab[i]; j < rangtab[i + 1]; j++)
            {
                i2irow[j] = i;
            }
            mean += nodes.at(i)->size;
            mini = min(mini, nodes.at(i)->size);
            maxi = max(maxi, nodes.at(i)->size);
        }
        printf("[%d] %d blocks, min size %d, mean size %f, max size %d\n", comm_rank(), nblk, mini, mean / nblk, maxi);
        // Compute elimination tree & neighbors
        map<Node*,set<int>> rows_tmp; // Global row/col
        for (int k = 0; k < nblk; k++) {
            rows_tmp[nodes.at(k).get()] = set<int>();
        }
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            vector<int> kcols(n->size);
            for (int i = 0; i < n->size; i++)
                kcols.at(i) = n->start + i;
            // Add local rows
            for (int j = n->start; j < n->end; j++)
            {
                for (SpMat::InnerIterator it(App, j); it; ++it)
                {
                    int i = it.row();
                    if (i >= n->end)
                    {
                        rows_tmp.at(n.get()).insert(i);
                    }
                }
            }
            vector<int> rows(rows_tmp.at(n.get()).begin(), rows_tmp.at(n.get()).end());
            sort(rows.begin(), rows.end());
            // Convert to neighbors
            set<int> nbrs_tmp;
            for (auto i : rows)
                nbrs_tmp.insert(i2irow(i));
            n->nbrs = vector<int>(nbrs_tmp.begin(), nbrs_tmp.end());
            sort(n->nbrs.begin(), n->nbrs.end());
            // Diagonal bloc
            blocs[{k, k}] = make_unique<Bloc>();
            auto &b = blocs.at({k, k});
            b->rows = kcols;
            b->cols = kcols;
            if(ij2rank({k,k}) == comm_rank()) {                
                b->allocate();
            }
            // Below-diagonal bloc
            for (auto nirow : n->nbrs)
            {                
                auto &nbr = nodes.at(nirow);
                blocs[{nirow, k}] = make_unique<Bloc>();
                auto &b = blocs.at({nirow, k});
                // Find rows
                auto lower = lower_bound(rows.begin(), rows.end(), nbr->start);
                auto upper = upper_bound(rows.begin(), rows.end(), nbr->end - 1);
                b->rows = vector<int>(lower, upper);                    
                b->cols = kcols;                
                if(ij2rank({nirow, k}) == comm_rank()) {                    
                    b->allocate();
                }
            }
            // Add to parent
            if (n->nbrs.size() > 0)
            {
                assert(n->nbrs.at(0) > k);
                int prow = n->nbrs.at(0); // parent in etree = first non zero in column
                n->parent = prow;
                auto &p = nodes.at(prow);
                for (auto i : rows_tmp.at(n.get()))
                {
                    if (i >= p->end)
                    {
                        rows_tmp.at(p.get()).insert(i);
                    }
                }
                p->children.push_back(k);
            }
            else
            {
                n->parent = -1;
            }
        }
        printf("[%d] Done allocating matrix\n", comm_rank());
        // Fill with A
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            for (int j = n->start; j < n->end; j++)
            {
                for (SpMat::InnerIterator it(App, j); it; ++it)
                {
                    int i = it.row();
                    if (i >= n->start)
                    {
                        // Find bloc
                        int irow = i2irow[i];
                        if(ij2rank({irow,k}) == comm_rank()) {
                            auto &b = blocs.at({irow,k});
                            auto found = lower_bound(b->rows.begin(), b->rows.end(), i);
                            assert(found != b->rows.end());
                            int ii = distance(b->rows.begin(), found);
                            int jj = j - n->start;
                            (*b->A())(ii, jj) = it.value();
                        }                        
                    }
                }
            }
        }
        // Count dependencies
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            for (auto i : n->nbrs)
            {
                for (auto j : n->nbrs)
                {
                    if (i >= j)
                    {
                        auto &b = blocs.at({i, j});
                        b->n_accumulate++;
                    }
                }
            }
        }
        printf("[%d] Done distributing matrix\n", comm_rank());
    }

    void print()
    {
        MatrixXd Aff = MatrixXd::Zero(perm.size(), perm.size());
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            int start = n->start;
            int size = n->size;
            Aff.block(start, start, size, size) = blocs.at({k, k})->A()->triangularView<Lower>();
            for (auto i : n->nbrs)
            {
                MatrixXd *Aik = blocs.at({i, k})->A();
                for (int ii = 0; ii < Aik->rows(); ii++)
                {
                    for (int jj = 0; jj < Aik->cols(); jj++)
                    {
                        Aff(blocs.at({i, k})->rows[ii], blocs.at({i, k})->cols[jj]) = (*Aik)(ii, jj);
                    }
                }
            }
        }
        cout << Aff << endl;
    }

    // Factor a diagonal pivot A[k,k] in-place
    void potf(int k)
    {
        MatrixXd *Ass = blocs.at({k,k})->A();
        timer t0 = wctime();
        int err = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', Ass->rows(), Ass->data(), Ass->rows());
        timer t1 = wctime();
        potf_us += (long long)(elapsed(t0, t1) * 1e6);
        assert(err == 0);
    }

    // Trsm a panel bloc A[i,j] in-place
    void trsm(int2 ij)
    {
        int i = ij[0];
        int j = ij[1];
        assert(i > j);
        MatrixXd *Ajj = blocs.at({j,j})->A();
        MatrixXd *Aij = blocs.at({i,j})->A();
        timer t0 = wctime();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    Aij->rows(), Aij->cols(), 1.0, Ajj->data(), Ajj->rows(), Aij->data(), Aij->rows());
        timer t1 = wctime();
        trsm_us += (long long)(elapsed(t0, t1) * 1e6);
    }

    // Perform a gemm between (i,k) and (j,k) and store the result at (i,j) in to_accumulate
    void gemm(int3 kijrow)
    {
        int krow = kijrow[0];
        int irow = kijrow[1];
        int jrow = kijrow[2];
        MatrixXd *Ais = blocs.at({irow, krow})->A();
        MatrixXd *Ajs = blocs.at({jrow, krow})->A();
        // Do the math
        timer t0, t1, t2;
        t0 = wctime();
        auto Aij_acc = make_unique<MatrixXd>(Ais->rows(), Ajs->rows());
        t1 = wctime();
        if (jrow == irow)
        { // Aii_ = -Ais Ais^T
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        Ais->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        else
        { // Aij_ = -Ais Ajs^T
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        Ais->rows(), Ajs->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), Ajs->data(), Ajs->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        t2 = wctime();
        {
            auto &mtx = blocs.at({irow, jrow})->to_accumulate_mtx;
            auto &acc = blocs.at({irow, jrow})->to_accumulate;
            lock_guard<mutex> lock(mtx);
            acc[krow] = move(Aij_acc);
        }
        allo_us += (long long)(elapsed(t0, t1) * 1e6);
        gemm_us += (long long)(elapsed(t1, t2) * 1e6);
    }

    void accumulate(int3 kijrow)
    {
        int krow = kijrow[0];
        int irow = kijrow[1];
        int jrow = kijrow[2];
        auto &mtx = blocs.at({irow, jrow})->to_accumulate_mtx;
        auto &acc = blocs.at({irow, jrow})->to_accumulate;
        {
            assert(!blocs.at({irow, jrow})->accumulating_busy.load());
            blocs.at({irow, jrow})->accumulating_busy.store(true);
        }
        unique_ptr<MatrixXd> Aij_acc;
        MatrixXd *Aij = blocs.at({irow, jrow})->A();
        timer t0, t1;
        {
            lock_guard<mutex> lock(mtx);
            Aij_acc = move(acc.at(krow));
            acc.erase(acc.find(krow));
        }
        t0 = wctime();
        if (jrow == irow)
        { // Aii_ = -Ais Ais^T
            auto I = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, jrow})->rows);
            for (int j = 0; j < Aij_acc->cols(); j++)
            {
                for (int i = j; i < Aij_acc->rows(); i++)
                {
                    (*Aij)(I[i], I[j]) += (*Aij_acc)(i, j);
                }
            }
        }
        else
        { // Aij_ = -Ais Ajs^T
            auto I = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, jrow})->rows);
            auto J = get_subids(blocs.at({jrow, krow})->rows, blocs.at({irow, jrow})->cols);
            for (int j = 0; j < Aij_acc->cols(); j++)
            {
                for (int i = 0; i < Aij_acc->rows(); i++)
                {
                    (*Aij)(I[i], J[j]) += (*Aij_acc)(i, j);
                }
            }
        }
        t1 = wctime();
        scat_us += (long long)(elapsed(t0, t1) * 1e6);
        {
            assert(blocs.at({irow, jrow})->accumulating_busy.load());
            blocs.at({irow, jrow})->accumulating_busy.store(false);
        }
    }

    void factorize(int n_threads)
    {
        printf("Rank %d starting %d threads\n", comm_rank(), n_threads);
        Logger log(1000000);
        Communicator comm(VERB);

        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(comm_rank()) + "]_");
        Taskflow<int>  pf(&tp, VERB);
        Taskflow<int2> tf(&tp, VERB);
        Taskflow<int3> gf(&tp, VERB);
        Taskflow<int3> rf(&tp, VERB);        

        auto am = comm.make_active_msg(
            [&](int &i, int &k, view<double> &Aik, view<int> &fulfill) {
                auto& b = this->blocs.at({i,k});
                b->allocate();
                memcpy(b->A()->data(), Aik.data(), Aik.size() * sizeof(double));
                for (auto j : fulfill)
                {   
                    if(i == k) {
                        tf.fulfill_promise({k,j});
                    } else {                    
                        gf.fulfill_promise(lower({k,i,j}));
                    }
                }
            });

        if (LOG > 0)
        {
            tp.set_logger(&log);
            comm.set_logger(&log);
        }

        /**
         * POTF needs to communicate its output to trigger TRSM
         * Input is assumed to be there and ready
         **/
        pf
            .set_mapping([&](int k) {
                assert(ij2rank({k,k}) == comm_rank());
                return (k % n_threads);
            })
            .set_indegree([&](int k) {
                assert(ij2rank({k,k}) == comm_rank());
                int nacc = n_to_accumulate({k, k});
                return nacc == 0 ? 1 : nacc ; // 1 (seeding) or # gemms before
            })
            .set_task([&](int k) {
                assert(accumulated({k, k}) == n_to_accumulate({k, k}));
                assert(ij2rank({k,k}) == comm_rank());
                potf(k);
            })
            .set_fulfill([&](int k) {
                assert(accumulated({k, k}) == n_to_accumulate({k, k}));
                assert(ij2rank({k,k}) == comm_rank());
                // Collect tasks and ranks to fulfill
                map<int,vector<int>> fulfills;
                auto &n = nodes.at(k);
                for(auto i: n->nbrs) {
                    int dest = ij2rank({i,k});
                    if(fulfills.count(dest) == 0) {
                        fulfills[dest] = {i};
                    } else {
                        fulfills[dest].push_back(i);
                    }
                }
                // Send data and fulfill dependency
                auto vAkk = view<double>(blocs.at({k,k})->A()->data(), blocs.at({k,k})->A()->size());
                for(auto dest_ff: fulfills) {
                    auto vff = view<int>(dest_ff.second.data(), dest_ff.second.size());
                    if(dest_ff.first == comm_rank()) { // Local: just ff
                        for(auto i: vff) {
                            tf.fulfill_promise({k,i});
                        }
                    } else { // Remote: send data & ff
                        am->send(dest_ff.first, k, k, vAkk, vff);
                    }
                }
            })
            .set_name([&](int k) {
                return "[" + to_string(comm_rank()) + "]_potf_" + to_string(k);
            })
            .set_priority([](int k) {
                return 3.0;
            });

        /**
         * TRSM need to communicate its output to trigger GEMMS
         * Inputs are assumed to be there and ready
         **/
        tf
            .set_mapping([&](int2 ki) {
                assert(ij2rank({ki[1], ki[0]}) == comm_rank());
                return (ki[0] % n_threads);
            })
            .set_indegree([&](int2 ki) {
                assert(ij2rank({ki[1], ki[0]}) == comm_rank());
                int k = ki[0];
                int i = ki[1];
                assert(i > k);
                return 1 + n_to_accumulate({i, k}); // above potf + # gemm before
            })
            .set_task([&](int2 ki) {
                assert(ij2rank({ki[1], ki[0]}) == comm_rank());
                assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));
                trsm(ki);
            })
            .set_fulfill([&](int2 ki) {
                assert(ij2rank({ki[1], ki[0]}) == comm_rank());
                assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));                
                // Collect tasks and ranks to fulfill
                int k = ki[0];
                int i = ki[1];
                auto& n = nodes.at(k);
                map<int,vector<int>> fulfills;
                for(auto j: n->nbrs) {
                    auto kij = lower({k,i,j});
                    int dest = ij2rank({kij[1], kij[2]});
                    if(fulfills.count(dest) == 0) {
                        fulfills[dest] = {j};
                    } else {
                        fulfills[dest].push_back(j);
                    }
                }
                // Send data and fulfill dependency
                auto vAik = view<double>(blocs.at({i,k})->A()->data(), blocs.at({i,k})->A()->size());
                for(auto dest_ff: fulfills) {
                    auto vff = view<int>(dest_ff.second.data(), dest_ff.second.size());
                    if(dest_ff.first == comm_rank()) { // Local: just ff
                        for(auto j: vff) {
                            gf.fulfill_promise(lower({k,i,j})); // gemms after
                        }
                    } else { // Remote: send data & ff
                        am->send(dest_ff.first, i, k, vAik, vff);
                    }
                }                
            })
            .set_name([&](int2 ki) {
                return "[" + to_string(comm_rank()) + "]_trsm_" + to_string(ki[0]) + "_" + to_string(ki[1]);
            })
            .set_priority([](int2 k) {
                return 2.0;
            });

        /**
         * GEMM is 100% local.
         * Both inputs are assumed to be on the node, and the output is at the same location, ready to be reduces
         **/
        gf
            .set_mapping([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                return (kij[0] % n_threads);
            })
            .set_indegree([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                int i = kij[1];
                int j = kij[2];
                assert(j <= i);
                return (i == j ? 1 : 2); // 1 potf or 2 trsms
            })
            .set_task([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                gemm(kij);
            })
            .set_fulfill([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                int k = kij[0];
                int i = kij[1];
                int j = kij[2];
                assert(k <= j);
                assert(j <= i);
                // printf("gf %d %d %d -> rf %d %d %d\n", comm_rank(), k, i, j, k, i, j);
                rf.fulfill_promise(kij); // The corresponding reduction
            })
            .set_name([&](int3 kij) {
                return "[" + to_string(comm_rank()) + "]_gemm_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
            })
            .set_priority([](int3 k) {
                return 1.0;
            });

        /**
         * REDUCTION is 100% local
         * Inputs are assumed to be there, and outputs is at the same location
         **/
        rf
            .set_mapping([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                return (kij[1] + kij[2]) % n_threads; // any i & j -> same thread. So k cannot appear in this expression
            })
            .set_indegree([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                return 1; // The corresponding gemm
            })
            .set_task([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                blocs.at({kij[1], kij[2]})->accumulated++;
                accumulate(kij);
            })
            .set_fulfill([&](int3 kij) {
                assert(ij2rank({kij[1], kij[2]}) == comm_rank());
                int i = kij[1];
                int j = kij[2];
                if (i == j)
                {
                    // printf("rf %d %d %d -> pf %d\n", k, i, j, i);
                    pf.fulfill_promise(i); // Corresponding potf
                }
                else
                {
                    // printf("rf %d %d %d -> tf %d %d\n", k, i, j, j, i);
                    tf.fulfill_promise({j, i}); // Corresponding trsm
                }
            })
            .set_name([&](int3 kij) {
                return "[" + to_string(comm_rank()) + "]_acc_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
            })
            .set_priority([](int3 k) {
                return 4.0;
            })
            .set_binding([](int3 k) {
                return true; // Bind task to thread
            });

        // // Start by seeding initial tasks
        MPI_Barrier(MPI_COMM_WORLD);
        timer t0 = wctime();
        for (int k = 0; k < nblk; k++)
        {
            if (ij2rank({k,k}) == comm_rank())
            {
                if (n_to_accumulate({k,k}) == 0)
                {
                    pf.fulfill_promise(k);
                }
            }
        }
        tp.join();
        printf("[%d] Tp & Comms done\n", comm_rank());
        MPI_Barrier(MPI_COMM_WORLD);
        timer t1 = wctime();
        printf("[%d] Factorization done, time %3.2e s.\n", comm_rank(), elapsed(t0, t1));
        printf("[%d] Potf %3.2e s., %3.2e s./thread\n", comm_rank(), double(potf_us / 1e6), double(potf_us / 1e6) / n_threads);
        printf("[%d] Trsm %3.2e s., %3.2e s./thread\n", comm_rank(), double(trsm_us / 1e6), double(trsm_us / 1e6) / n_threads);
        printf("[%d] Gemm %3.2e s., %3.2e s./thread\n", comm_rank(), double(gemm_us / 1e6), double(gemm_us / 1e6) / n_threads);
        printf("[%d] Allo %3.2e s., %3.2e s./thread\n", comm_rank(), double(allo_us / 1e6), double(allo_us / 1e6) / n_threads);
        printf("[%d] Scat %3.2e s., %3.2e s./thread\n", comm_rank(), double(scat_us / 1e6), double(scat_us / 1e6) / n_threads);
        printf(">>>>%d,%d,%d,%3.2e\n", comm_rank(), comm_size(), n_threads, elapsed(t0, t1));

        auto am_send_block = comm.make_active_msg(
            [&](int& i, int &k, view<double> &Aik) {
                auto &b = this->blocs.at({i, k});
                b->allocate();                
                memcpy(b->A()->data(), Aik.data(), Aik.size() * sizeof(double));
            });        

        // Exchange data back to process 0 for solve
        if (comm_rank() != 0)
        {
            for (int k = 0; k < nblk; k++)
            {                
                auto &n = nodes.at(k);
                if(ij2rank({k,k}) == comm_rank()) { // Pivot
                    auto *Akk = blocs.at({k,k})->A();
                    auto vAkk = view<double>(Akk->data(), Akk->size());
                    am_send_block->blocking_send(0, k, k, vAkk);
                }
                for (auto i : n->nbrs)
                {
                    if(ij2rank({i,k}) == comm_rank()) { // Panel
                        MatrixXd *Aik = blocs.at({i,k})->A();
                        auto vAik = view<double>(Aik->data(), Aik->size());
                        am_send_block->blocking_send(0, i, k, vAik);
                    }
                }
            }
        }
        else
        {
            for (int k = 0; k < nblk; k++)
            {
                auto &n = nodes.at(k);
                if(ij2rank({k,k}) != comm_rank()) {
                    comm.recv_process();
                }
                for (auto i : n->nbrs)
                {
                    if(ij2rank({i,k}) != comm_rank()) { 
                        comm.recv_process();
                    }
                }
            }
        }

        if (LOG > 0)
        {
            ofstream logfile;
            string filename = FOLDER + "/snchol_" + to_string(comm_size()) + "_" + to_string(n_threads) + "_" + to_string(App.rows()) + ".log." + to_string(comm_rank());
            printf("[%d] Logger saved to %s\n", comm_rank(), filename.c_str());
            logfile.open(filename);
            logfile << log;
            logfile.close();
        }
    }

    VectorXd solve(VectorXd &b)
    {
        assert(comm_rank() == 0);
        VectorXd xglob = perm.asPermutation() * b;
        // Set solution on each node
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            k->xsol = xglob.segment(k->start, k->size);
        }
        // Forward
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            // Pivot xs <- Lss^-1 xs
            MatrixXd *Lss = blocs.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocs.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // xn = -Lns xs
                cblas_dgemv(CblasColMajor, CblasNoTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), k->xsol.data(), 1, 0.0, xn.data(), 1);
                // Reduce into xn
                auto I = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    n->xsol(I[i]) += xn(i);
                }
            }
        }
        // Backward
        for (int krow = nblk - 1; krow >= 0; krow--)
        {
            auto &k = nodes.at(krow);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocs.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // Fetch from xn
                auto I = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    xn(i) = n->xsol(I[i]);
                }
                // xs -= Lns^T xn
                cblas_dgemv(CblasColMajor, CblasTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), xn.data(), 1, 1.0, k->xsol.data(), 1);
            }
            // xs = Lss^-T xs
            MatrixXd *Lss = blocs.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
        }
        // Back to x
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            xglob.segment(k->start, k->size) = k->xsol;
        }
        return perm.asPermutation().transpose() * xglob;
    }
};

TEST(snchol, one)
{
    printf("[%d] Hello from %s\n", comm_rank(), processor_name().c_str());
    DistMat dm(FILENAME, N_LEVELS, BLOCK_SIZE, PROWS, PCOLS);
    dm.factorize(N_THREADS);
    // Testing
    if (comm_rank() == 0)
    {
        SpMat A = dm.A;
        VectorXd b = random(A.rows(), 2019);
        VectorXd x = dm.solve(b);
        double res = (A * x - b).norm() / b.norm();
        ASSERT_LE(res, 1e-12);
        printf("|Ax-b|/|b| = %e\n", res);
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    int err = MPI_Init_thread(NULL, NULL, req, &prov);
    assert(err == 0 && prov == req);
    ::testing::InitGoogleTest(&argc, argv);
    if (argc >= 2) {
        PROWS = atoi(argv[1]);
    }
    if (argc >= 3) {
        PCOLS = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        FILENAME = argv[3];
    }
    if (argc >= 5)
    {
        N_LEVELS = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        N_THREADS = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        VERB = atoi(argv[6]);
    }
    if (argc >= 8)
    {
        BLOCK_SIZE = atoi(argv[7]);
    }
    if (argc >= 9)
    {
        LOG = atoi(argv[8]);
    }
    if (argc >= 10)
    {
        FOLDER = argv[9];
    }
    const int return_flag = RUN_ALL_TESTS();
    MPI_Finalize();
    return return_flag;
}
