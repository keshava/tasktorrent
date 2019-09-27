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

int3 lower(int3 ijk)
{
    int i = ijk[0];
    int j = ijk[1];
    int k = ijk[2];
    if (i >= j)
        return ijk;
    else
        return {j,i,k};
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

    // Solutions
    mutex xsol_mtx;             // Protect accumulations into xsol. This avoid an extra annoying TF
    VectorXd xsol;              // Solution on this node. 
                                // For diagonal bloc A[i,i] -> the solution x[i]
                                // For off-diagonal bloc A[i,j] -> the partial x[i] to be accumulated into x[i] eventually
    
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
    int start;             // Global row/col
    int size;              // Global row/col
    int end;               // Global row/col
    vector<int> nbrs;      // Global node id
    vector<int> ancestors; // Global node id

    // Ordering
    vector<int> children; // Global node id    
    int parent;           // Global node id

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
        assert(i >= j);
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
            blocs[{k,k}] = make_unique<Bloc>();
            auto &b = blocs.at({k,k});
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
            // Add to parent & before
            for(auto nb: n->nbrs) {
                nodes.at(nb)->ancestors.push_back(k);
            }
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
            Aff.block(start, start, size, size) = blocs.at({k,k})->A()->triangularView<Lower>();
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

    // Perform a gemm between A[i,k] and A[j,k] and store the result at A[i,j] to be accumulated later
    void gemm(int3 ijk)
    {
        int i = ijk[0];
        int j = ijk[1];
        int k = ijk[2];        
        assert(i >= j);
        assert(j >  k);
        MatrixXd *Aik = blocs.at({i,k})->A();
        MatrixXd *Ajk = blocs.at({j,k})->A();
        assert(Aik->cols() == Ajk->cols());
        // Do the math
        timer t0, t1, t2;
        t0 = wctime();
        auto Aij_acc = make_unique<MatrixXd>(Aik->rows(), Ajk->rows());
        t1 = wctime();
        if (j == i)
        { // Aii_ = -Aik Aik^T
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        Aik->rows(), Aik->cols(), -1.0, Aik->data(), Aik->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        else
        { // Aij_ = -Aik Ajk^T
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        Aik->rows(), Ajk->rows(), Aik->cols(), -1.0, Aik->data(), Aik->rows(), Ajk->data(), Ajk->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        t2 = wctime();
        {
            auto &mtx = blocs.at({i,j})->to_accumulate_mtx;
            auto &acc = blocs.at({i,j})->to_accumulate;
            lock_guard<mutex> lock(mtx);
            acc[k] = move(Aij_acc);
        }
        allo_us += (long long)(elapsed(t0, t1) * 1e6);
        gemm_us += (long long)(elapsed(t1, t2) * 1e6);
    }

    void accumulate(int3 ijk)
    {
        int i = ijk[0];
        int j = ijk[1];
        int k = ijk[2];        
        assert(i >= j);
        assert(j >  k);
        auto &mtx = blocs.at({i,j})->to_accumulate_mtx;
        auto &acc = blocs.at({i,j})->to_accumulate;
        {
            assert(!blocs.at({i,j})->accumulating_busy.load());
            blocs.at({i,j})->accumulating_busy.store(true);
        }
        unique_ptr<MatrixXd> Aij_acc;
        MatrixXd *Aij = blocs.at({i,j})->A();
        timer t0, t1;
        {
            lock_guard<mutex> lock(mtx);
            Aij_acc = move(acc.at(k));
            acc.erase(acc.find(k));
        }
        t0 = wctime();
        if (j == i)
        { // Aii_ = -Aik Aik^T
            auto I = get_subids(blocs.at({i,k})->rows, blocs.at({i,j})->rows);
            for (int j = 0; j < Aij_acc->cols(); j++)
            {
                for (int i = j; i < Aij_acc->rows(); i++)
                {
                    (*Aij)(I[i], I[j]) += (*Aij_acc)(i, j);
                }
            }
        }
        else
        { // Aij_ = -Aik Ajk^T
            auto I = get_subids(blocs.at({i,k})->rows, blocs.at({i,j})->rows);
            auto J = get_subids(blocs.at({j,k})->rows, blocs.at({i,j})->cols);
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
            assert(blocs.at({i,j})->accumulating_busy.load());
            blocs.at({i,j})->accumulating_busy.store(false);
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
                        tf.fulfill_promise({j,k});
                    } else {
                        gf.fulfill_promise(lower({i,j,k}));
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
                int nacc = n_to_accumulate({k,k});
                return nacc == 0 ? 1 : nacc ; // 1 (seeding) or # gemms before
            })
            .set_task([&](int k) {
                assert(accumulated({k,k}) == n_to_accumulate({k,k}));
                assert(ij2rank({k,k}) == comm_rank());
                potf(k);
            })
            .set_fulfill([&](int k) {
                assert(accumulated({k,k}) == n_to_accumulate({k,k}));
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
                            tf.fulfill_promise({i,k});
                        }
                    } else { // Remote: send data & ff
                        am->send(dest_ff.first, k,k, vAkk, vff);
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
            .set_mapping([&](int2 ik) {
                assert(ij2rank(ik) == comm_rank());
                return (ik[1] % n_threads);
            })
            .set_indegree([&](int2 ik) {
                assert(ij2rank(ik) == comm_rank());
                int i = ik[0];
                int k = ik[1];
                assert(i > k);
                return 1 + n_to_accumulate({i,k}); // above potf + # gemm before
            })
            .set_task([&](int2 ik) {
                assert(ij2rank(ik) == comm_rank());
                assert(accumulated(ik) == n_to_accumulate(ik));
                trsm(ik);
            })
            .set_fulfill([&](int2 ik) {
                assert(ij2rank(ik) == comm_rank());
                assert(accumulated(ik) == n_to_accumulate(ik));
                // Collect tasks and ranks to fulfill                
                int i = ik[0];
                int k = ik[1];
                auto& n = nodes.at(k);
                map<int,vector<int>> fulfills;
                for(auto j: n->nbrs) {
                    auto gijk = lower({i,j,k});
                    int dest = ij2rank({gijk[0], gijk[1]});
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
                            gf.fulfill_promise(lower({i,j,k})); // gemms after
                        }
                    } else { // Remote: send data & ff
                        am->send(dest_ff.first, i, k, vAik, vff);
                    }
                }                
            })
            .set_name([&](int2 ik) {
                return "[" + to_string(comm_rank()) + "]_trsm_" + to_string(ik[0]) + "_" + to_string(ik[1]);
            })
            .set_priority([](int2 ik) {
                return 2.0;
            });

        /**
         * GEMM is 100% local.
         * Both inputs are assumed to be on the node, and the output is at the same location, ready to be reduces
         **/
        gf
            .set_mapping([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                return (ijk[2] % n_threads);
            })
            .set_indegree([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                int i = ijk[0];
                int j = ijk[1];
                assert(i >= j);
                return (i == j ? 1 : 2); // 1 potf or 2 trsms
            })
            .set_task([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                gemm(ijk);
            })
            .set_fulfill([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                int i = ijk[0];
                int j = ijk[1];
                int k = ijk[2];
                assert(i >= j);
                assert(j >  k);
                rf.fulfill_promise(ijk); // The corresponding reduction
            })
            .set_name([&](int3 ijk) {
                return "[" + to_string(comm_rank()) + "]_gemm_" + to_string(ijk[0]) + "_" + to_string(ijk[1]) + "_" + to_string(ijk[2]);
            })
            .set_priority([](int3 ijk) {
                return 1.0;
            });

        /**
         * REDUCTION is 100% local
         * Inputs are assumed to be there, and outputs is at the same location
         **/
        rf
            .set_mapping([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                return (ijk[0] + ijk[1]) % n_threads; // any i & j -> same thread. So k cannot appear in this expression
            })
            .set_indegree([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                return 1; // The corresponding gemm
            })
            .set_task([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                blocs.at({ijk[0], ijk[1]})->accumulated++;
                accumulate(ijk);
            })
            .set_fulfill([&](int3 ijk) {
                assert(ij2rank({ijk[0], ijk[1]}) == comm_rank());
                int i = ijk[0];
                int j = ijk[1];
                if (i == j)
                {
                    pf.fulfill_promise(i); // Corresponding potf
                }
                else
                {
                    tf.fulfill_promise({i,j}); // Corresponding trsm
                }
            })
            .set_name([&](int3 ijk) {
                return "[" + to_string(comm_rank()) + "]_acc_" + to_string(ijk[0]) + "_" + to_string(ijk[1]) + "_" + to_string(ijk[2]);
            })
            .set_priority([](int3 ijk) {
                return 4.0;
            })
            .set_binding([](int3 ijk) {
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

        // auto am_send_block = comm.make_active_msg(
        //     [&](int& i, int &j, view<double> &Aij) {
        //         auto &b = this->blocs.at({i,j});
        //         b->allocate();                
        //         memcpy(b->A()->data(), Aij.data(), Aij.size() * sizeof(double));
        //     });        

        // Exchange data back to process 0 for solve
        // if (comm_rank() != 0)
        // {
        //     for (int k = 0; k < nblk; k++)
        //     {                
        //         auto &n = nodes.at(k);
        //         if(ij2rank({k,k}) == comm_rank()) { // Pivot
        //             auto *Akk = blocs.at({k,k})->A();
        //             auto vAkk = view<double>(Akk->data(), Akk->size());
        //             am_send_block->blocking_send(0, k,k, vAkk);
        //         }
        //         for (auto i : n->nbrs)
        //         {
        //             if(ij2rank({i,k}) == comm_rank()) { // Panel
        //                 MatrixXd *Aik = blocs.at({i,k})->A();
        //                 auto vAik = view<double>(Aik->data(), Aik->size());
        //                 am_send_block->blocking_send(0, i, k, vAik);
        //             }
        //         }
        //     }
        // }
        // else
        // {
        //     for (int k = 0; k < nblk; k++)
        //     {
        //         auto &n = nodes.at(k);
        //         if(ij2rank({k,k}) != comm_rank()) {
        //             comm.recv_process();
        //         }
        //         for (auto i : n->nbrs)
        //         {
        //             if(ij2rank({i,k}) != comm_rank()) { 
        //                 comm.recv_process();
        //             }
        //         }
        //     }
        // }

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

    // Overwrite bjj->xsol by Ljj^-1 bjj->xsol
    void fwd_diag(int j) {        
        // Pivot x[j] = L[j,j]^-1 x[j]
        auto &bjj = blocs.at({j,j});
        MatrixXd *Ljj = bjj->A();
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, Ljj->rows(), Ljj->data(), Ljj->rows(), bjj->xsol.data(), 1);
    }

    // Overwrite bij->xsol by - Lij bjj->xsol
    void fwd_panel(int2 ij) {
        // Partial panel x[i] = - L[i,j] x[j]        
        int j = ij[1];
        auto &bjj = blocs.at({j,j});
        auto &bij = blocs.at(ij);
        MatrixXd *Lij = bij->A();
        assert(bij->rows.size() == Lij->rows());
        VectorXd xitmp = VectorXd::Zero(Lij->rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, Lij->rows(), Lij->cols(), -1.0, Lij->data(), Lij->rows(), bjj->xsol.data(), 1, 0.0, xitmp.data(), 1);
        bij->xsol = xitmp;
    }

    // + Reduce bij->xsol into bii->xsol
    void fwd_reduction(int2 ij) {
        // Reduce partial x at [i,j] into x at [i,i]
        int i = ij[0];
        auto &bii = blocs.at({i,i});
        auto &bij = blocs.at(ij);
        auto I = get_subids(bij->rows, bii->rows);
        assert(bij->xsol.size() == bij->rows.size());
        assert(bii->xsol.size() == bii->rows.size());
        lock_guard<mutex> lock(bii->xsol_mtx);
        for (int i = 0; i < bij->xsol.size(); i++)
        {
            bii->xsol(I[i]) += bij->xsol(i);
        }
    }

    // Overwrite bii->xsol by Lii^T bii->xsol
    void bwd_diag(int k) {
        // Pivot x[k] = Lkk^-T x[k]
        auto &bkk = blocs.at({k,k});
        MatrixXd *Lkk = bkk->A();
        cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, Lkk->rows(), Lkk->data(), Lkk->rows(), bkk->xsol.data(), 1);
    }

    // Overwrite bij->xsol by - Lij^T bii->xsol
    void bwd_panel(int2 ij) {
        // Partial panel x[j] -= Lij^T xn
        int i = ij[0];
        auto &bii = blocs.at({i,i});
        auto &bij = blocs.at(ij);
        auto I = get_subids(bij->rows, bii->rows);
        MatrixXd *Lij = bij->A();        
        VectorXd xjtmp = VectorXd::Zero(Lij->cols());
        VectorXd xitmp = VectorXd::Zero(Lij->rows());        
        for (int i = 0; i < bij->rows.size(); i++) {
            xitmp[i] = bii->xsol(I[i]);
        }
        cblas_dgemv(CblasColMajor, CblasTrans, Lij->rows(), Lij->cols(), -1.0, Lij->data(), Lij->rows(), xitmp.data(), 1, 1.0, xjtmp.data(), 1);
        bij->xsol = xjtmp;
        assert(bij->xsol.size() == Lij->cols());
    }

    void bwd_reduction(int2 ij) {
        int j = ij[1];
        auto &bjj = blocs.at({j,j});
        auto &bij = blocs.at(ij);
        assert(bjj->xsol.size() == bjj->rows.size());
        assert(bjj->xsol.size() == bij->xsol.size());
        lock_guard<mutex> lock(bjj->xsol_mtx);
        for (int i = 0; i < bij->xsol.size(); i++)
        {
            bjj->xsol(i) += bij->xsol(i);
        }
    }

    VectorXd solve(VectorXd &b, int n_threads)
    {
        VectorXd xglob = perm.asPermutation() * b;
        VectorXd xglob_sol = VectorXd::Zero(xglob.size());
        // Set solution on each diagonal node
        for (int k = 0; k < nblk; k++)
        {
            auto &b = blocs.at({k,k});
            auto &n = nodes.at(k);
            if(ij2rank({k,k}) == comm_rank()) {                
                b->xsol = xglob.segment(n->start, n->size);
            } else {
                b->xsol = VectorXd::Zero(n->size);
            }
        }

        printf("Rank %d starting %d threads\n", comm_rank(), n_threads);
        Logger log(1000000);
        Communicator comm(VERB);
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(comm_rank()) + "]_solve_");
        Taskflow<int>  fwd_df(&tp, VERB);
        Taskflow<int2> fwd_pf(&tp, VERB);
        Taskflow<int2> bwd_pf(&tp, VERB);
        Taskflow<int>  bwd_df(&tp, VERB);

        /**
         * FORWARD
         **/

        auto am_fwd_diag = comm.make_active_msg(
            [&](int &j, view<double> &xj, view<int> &ff) {
                auto& bjj = this->blocs.at({j,j});
                bjj->xsol = VectorXd::Zero(xj.size());
                memcpy(bjj->xsol.data(), xj.data(), xj.size() * sizeof(double));
                for(auto i: ff) {
                    fwd_pf.fulfill_promise({i,j});
                }
            });

        auto am_fwd_panel = comm.make_active_msg(
            [&](int &i, int &j, view<double> &xij) {
                auto& bij = this->blocs.at({i,j});
                bij->xsol = VectorXd::Zero(xij.size());
                memcpy(bij->xsol.data(), xij.data(), xij.size() * sizeof(double));
                fwd_reduction({i,j});
                fwd_df.fulfill_promise(i);
            });        

        fwd_df
            .set_mapping([&](int j) {
                assert(ij2rank({j,j}) == comm_rank());
                return (j % n_threads);
            })
            .set_indegree([&](int j) {
                assert(ij2rank({j,j}) == comm_rank());
                int nanc = nodes.at(j)->ancestors.size();
                return nanc == 0 ? 1 : nanc ;
            })
            .set_task([&](int j) {
                assert(ij2rank({j,j}) == comm_rank());
                fwd_diag(j);
            })
            .set_fulfill([&](int j) {
                assert(ij2rank({j,j}) == comm_rank());
                auto& bjj = blocs.at({j,j});
                auto& n = nodes.at(j);
                if(n->nbrs.size() == 0) {
                    // Trigger backward
                    bwd_df.fulfill_promise(j);
                } else {
                    // Trigger forward panel under pivot
                    // Collect tasks and ranks to fulfill
                    map<int,vector<int>> fulfills;                    
                    for(auto i: n->nbrs) {
                        int dest = ij2rank({i,j});
                        if(fulfills.count(dest) == 0) {
                            fulfills[dest] = {i};
                        } else {
                            fulfills[dest].push_back(i);
                        }
                    }
                    // Send data and fulfill dependency
                    auto vxsol = view<double>(bjj->xsol.data(), bjj->xsol.size());
                    for(auto dest_ff: fulfills) {
                        auto vff = view<int>(dest_ff.second.data(), dest_ff.second.size());
                        if(dest_ff.first == comm_rank()) { // Local: just ff
                            for(auto i: vff) {
                                fwd_pf.fulfill_promise({i,j});
                            }
                        } else { // Remote: send data & ff
                            am_fwd_diag->send(dest_ff.first,j,vxsol,vff);
                        }
                    }
                }                
            })
            .set_name([&](int j) {
                return "[" + to_string(comm_rank()) + "]_fwd_diag_" + to_string(j);
            });

        fwd_pf
            .set_mapping([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                return (ij[0] % n_threads);
            })
            .set_indegree([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                return 1;
            })
            .set_task([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                fwd_panel(ij);
            })
            .set_fulfill([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                int i = ij[0];
                int j = ij[1];
                int dest = ij2rank({i,i});
                auto &bij = blocs.at(ij);
                if(dest == comm_rank()) {
                    fwd_reduction(ij);
                    fwd_df.fulfill_promise(i);
                } else {
                    auto vxsol = view<double>(bij->xsol.data(), bij->xsol.size());
                    am_fwd_panel->send(dest, i, j, vxsol);
                }                
            })
            .set_name([&](int2 ij) {
                return "[" + to_string(comm_rank()) + "]_fwd_panel_" + to_string(ij[0]) + "_" + to_string(ij[1]);;
            });

        /**
         * BACKWARD
         */

        auto am_bwd_diag = comm.make_active_msg(
            [&](int &i, view<double> &xi, view<int> &ff) {
                auto& bii = this->blocs.at({i,i});
                bii->xsol = VectorXd::Zero(xi.size());
                memcpy(bii->xsol.data(), xi.data(), xi.size() * sizeof(double));
                for(auto j: ff) {
                    bwd_pf.fulfill_promise({i,j});
                }
            });

        auto am_bwd_panel = comm.make_active_msg(
            [&](int &i, int &j, view<double> &xij) {
                auto& bij = this->blocs.at({i,j});
                bij->xsol = VectorXd::Zero(xij.size());
                memcpy(bij->xsol.data(), xij.data(), xij.size() * sizeof(double));
                this->bwd_reduction({i,j});
                bwd_df.fulfill_promise(j);
            });

        auto am_send_sol = comm.make_active_msg(
            [&](int &j, view<double> &xj) {
                int start = nodes.at(j)->start;
                for(int i = 0; i < xj.size(); i++) {
                    xglob_sol[start + i] = xj.data()[i];
                }
            });

        bwd_df
            .set_mapping([&](int i) {
                assert(ij2rank({i,i}) == comm_rank());
                return (i % n_threads);
            })
            .set_indegree([&](int i) {
                assert(ij2rank({i,i}) == comm_rank());
                int nnbrs = nodes.at(i)->nbrs.size();
                return nnbrs == 0 ? 1 : nnbrs ; // If no neighbors, its triggered by fwd diag ; otherwise, by its nbrs
            })
            .set_task([&](int i) {
                assert(ij2rank({i,i}) == comm_rank());
                bwd_diag(i);
            })
            .set_fulfill([&](int i) {
                assert(ij2rank({i,i}) == comm_rank());
                auto& bii = blocs.at({i,i});
                auto &n = nodes.at(i);
                // Collect tasks and ranks to fulfill
                map<int,vector<int>> fulfills;                
                for(auto j: n->ancestors) {
                    int dest = ij2rank({i,j});
                    if(fulfills.count(dest) == 0) {
                        fulfills[dest] = {j};
                    } else {
                        fulfills[dest].push_back(j);
                    }
                }
                // Send data and fulfill dependency
                auto vxsol = view<double>(bii->xsol.data(), bii->xsol.size());
                for(auto dest_ff: fulfills) {
                    auto vff = view<int>(dest_ff.second.data(), dest_ff.second.size());
                    if(dest_ff.first == comm_rank()) { // Local: just ff
                        for(auto j: vff) {                        
                            bwd_pf.fulfill_promise({i,j});
                        }
                    } else { // Remote: send data & ff
                        am_bwd_diag->send(dest_ff.first,i,vxsol,vff);
                    }
                }
                // Write solution to xglob_sol
                if(comm_rank() == 0) {                    
                    xglob_sol.segment(n->start, n->size) = bii->xsol;
                } else {
                    am_send_sol->send(0, i, vxsol);
                }
            })
            .set_name([&](int j) {
                return "[" + to_string(comm_rank()) + "]_bwd_diag_" + to_string(j);
            });

        bwd_pf
            .set_mapping([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                return (ij[0] % n_threads);
            })
            .set_indegree([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                return 1;
            })
            .set_task([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                bwd_panel(ij);
            })
            .set_fulfill([&](int2 ij) {
                assert(ij2rank(ij) == comm_rank());
                int i = ij[0];
                int j = ij[1];
                int dest = ij2rank({j,j});
                auto &bij = blocs.at(ij);
                if(dest == comm_rank()) {
                    bwd_reduction(ij);
                    bwd_df.fulfill_promise(j);
                } else {
                    auto vxsol = view<double>(bij->xsol.data(), bij->xsol.size());
                    am_bwd_panel->send(dest, i, j, vxsol);
                }   
            })
            .set_name([&](int2 ij) {
                return "[" + to_string(comm_rank()) + "]_bwd_panel_" + to_string(ij[0]) + "_" + to_string(ij[1]);;
            });

        // Start by seeding initial tasks
        MPI_Barrier(MPI_COMM_WORLD);
        timer t0 = wctime();
        for (int k = 0; k < nblk; k++)
        {
            if (ij2rank({k,k}) == comm_rank())
            {
                if (nodes.at(k)->ancestors.size() == 0)
                {
                    fwd_df.fulfill_promise(k);
                }
            }
        }
        tp.join();
        MPI_Barrier(MPI_COMM_WORLD);
        timer t1 = wctime();
        printf("[%d] Solve done, time %3.2e s.\n", comm_rank(), elapsed(t0, t1));
        return perm.asPermutation().transpose() * xglob_sol;
    }
};

TEST(snchol, one)
{
    printf("[%d] Hello from %s\n", comm_rank(), processor_name().c_str());
    assert(PROWS * PCOLS == comm_size());
    DistMat dm(FILENAME, N_LEVELS, BLOCK_SIZE, PROWS, PCOLS);
    SpMat A = dm.A;
    dm.factorize(N_THREADS);
    VectorXd b = random(A.rows(), 2019);
    VectorXd x = dm.solve(b, N_THREADS);
    // Testing
    if (comm_rank() == 0)
    {        
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
