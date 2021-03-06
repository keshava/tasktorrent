#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"
#include "complex.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;


/*
Parametrized priorities for cholesky:
0. No priority, only enforces potrf>trsm>gemm
1. Row-based priority, prioritize tasks with smaller row number in addition to priority 0.
2. Critical path priority, prioritize tasks with longest distance to the exit task. For references, check out the paper
    Beaumont, Olivier, et al. "A Makespan Lower Bound for the Scheduling of the Tiled Cholesky Factorization based on ALAP Schedule." (2020).
3. Critical path and row priority, prioritize tasks with smaller row number in addition to priority 2. We also enforces potrf>trsm>gemm
*/

enum PrioKind { no = 0, row = 1, cp = 2, cp_row = 3};

void cholesky(const int n_threads, const int verb, const int n, const int nb, const int nprows, const int npcols, 
              const PrioKind prio_kind, const bool log, const bool deps_log, const bool test, const int accumulate_parallel)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    assert(nprows * npcols == n_ranks);
    std::atomic<long long int> potrf_us_t(0);
    std::atomic<long long int> trsm_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    // Initializes the matrix
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd A;
    A = MatrixXd::NullaryExpr(n*nb,n*nb,val);
    MatrixXd L = A;
    vector<unique_ptr<MatrixXd>> blocks(nb*nb);
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            blocks[ii+jj*nb]=make_unique<MatrixXd>(n,n);
            *blocks[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
        }
    }

    // Holds the temporary matrices result of gemm to be accumulated by accu
    // Each block holds data to be accumulated into a given block[ii+jj*nb]
    struct acc_data {
        std::map<int, std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
        std::mutex mtx; // Protects that map
    };
    std::vector<acc_data> gemm_results(nb*nb); // gemm_results[ii+jj*nb] holds the data to be accumulated into blocks[ii+jj*nb]

    // Map tasks to ranks
    auto block_2_rank = [&](int i, int j) {
        int r = (j % npcols) * nprows + (i % nprows);
        assert(r >= 0 && r < n_ranks);
        return r;
    };

    // Map threads to ranks
    auto block_2_thread = [&](int i, int j) {
        int ii = i / nprows;
        int jj = j / npcols;
        int nbit = nb / nprows;
        return (ii + jj * nbit) % n_threads;
    };

    // Set priorities
    auto potf_block_2_prio = [&](int j) {
        if (prio_kind == PrioKind::cp_row) {
            return (double)(9*(nb - j)-1) + 18 * nb * nb;
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*(nb - j)-1);
        } 
        else if(prio_kind == PrioKind::row) {
            return 3.0*(double)(nb-j);
        } 
        else {
            return 3.0;
        }
    };
    auto trsm_block_2_prio = [&](int2 ij) {
        if (prio_kind == PrioKind::cp_row) {
            return (double)((nb - ij[0]) + nb * (9.0 * nb - 9.0 * ij[1] - 2.0) + 9 * nb * nb);
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*(nb - ij[1])-2);
        } 
        else if(prio_kind == PrioKind::row) {
            return 2.0*(double)(nb - ij[0]);
        } 
        else {
            return 2.0;
        }
    };
    auto gemm_block_2_prio = [&](int3 kij) {
        if (prio_kind == PrioKind::cp_row) {
            if (accumulate_parallel) {
                return (double)(nb - kij[1]) + nb * (9.0 * nb - 9.0 * kij[2] - 2.0);
            }
            else {
                return (double)(nb - kij[1]) + nb * (9.0 * nb - 3.0 * kij[2] - 6.0 * kij[0] - 2.0);
            }
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*nb-9*kij[2]-2);
        } 
        else if(prio_kind == PrioKind::row) {
            return (double)(nb - kij[1]);
        } 
        else {
            return 1.0;
        }
    };
    // Names
    auto potrf_name = [](int j, int r) {
        return "POTRF_" + to_string(j) + "_r" + to_string(r);
    };
    auto trsm_name = [](int2 ij, int r) {
        return "TRSM_" + to_string(ij[0]) + "_" + to_string(ij[1]) + "_r" + to_string(r);
    };
    auto gemm_name = [](int3 kij, int r) {
        return "GEMM_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_r" + to_string(r);
    };
    auto accu_name = [](int3 kij, int r) {
        return "ACCU_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_r" + to_string(r);
    };

    const int nbmax = 15;
    MPI_Barrier(MPI_COMM_WORLD);
    if(comm_rank() == 0) {
        printf("Block -> Rank\n");
        for(int i = 0; i < min(nbmax, nb); i++) {
            for(int j = 0; j < min(nbmax, nb); j++) {
                if(i >= j) {
                    printf("%2d ", block_2_rank(i, j));
                }
            }
            printf("\n");
        }
        printf("Potf/trsm -> Priority\n");
        for(int i = 0; i < min(nbmax, nb); i++) {
            for(int j = 0; j < min(nbmax, nb); j++) {
                if(i == j) {
                    printf("%5f ", potf_block_2_prio(i));
                } else if (i > j) {
                    printf("%5f ", trsm_block_2_prio({i,j}));
                };
            }
            printf("\n");
        }
        printf("Gemm -> Priority\n");
        for(int k = 0; k < min(nbmax, nb); k++) {
            printf("k = %d\n", k);
            for(int i = 0; i < min(nbmax, nb); i++) {
                for(int j = 0; j < min(nbmax, nb); j++) {
                    if(i >= j) {
                        if(k < j) {
                            printf("%5f ", gemm_block_2_prio({k,i,j}));
                        } else {
                            printf(".     ");
                        }
                    }
                }
                printf("\n");
            }
        }
    }
    for(int r = 0; r < ttor::comm_size(); r++) {
        if(r == comm_rank()) {
            printf("[%d] Block -> thread\n", r);
            for(int i = 0; i < min(nbmax, nb); i++) {
                for(int j = 0; j < min(nbmax, nb); j++) {
                    if(i >= j && block_2_rank(i,j) == r) {
                        printf("%2d ", block_2_thread(i, j));
                    } else {
                        printf(" . ");
                    }
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "Wk_Chol_" + to_string(rank) + "_");
    Taskflow<int>  potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);
    Taskflow<int3> accu(&tp, verb);

    Logger logger(1000000);
    if(log) {
        tp.set_logger(&logger);
        comm.set_logger(&logger);
    }

    DepsLogger dlog(1000000);

    // Send a potrf'ed pivot A(k,k) and trigger trsms below requiring A(k,k)
    auto am_trsm = comm.make_active_msg( 
            [&](view<double> &Ljj, int& j, view<int>& is) {
                *blocks[j+j*nb] = Map<MatrixXd>(Ljj.data(), n, n);
                for(auto& i: is) {
                    trsm.fulfill_promise({i,j});
                }
            });

    /**
     * j is the pivot's position at A(j,j)
     */
    potrf.set_task([&](int j) { // A[j,j] -> A[j,j]
            assert(block_2_rank(j,j) == rank);
            timer t_ = wctime();
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, blocks[j+j*nb]->data(), n);
            timer t__ = wctime();
            potrf_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int j) { // Triggers all trsms on rows i > j, A[i,j]
            assert(block_2_rank(j,j) == rank);
            vector<vector<int>> fulfill(n_ranks);
            for (int i = j+1; i<nb; i++) {
                fulfill[block_2_rank(i,j)].push_back(i);
            }
            for (int r = 0; r < n_ranks; r++) { // Looping through all outgoing dependency edges
                if (rank == r) {
                    for (auto& i: fulfill[r]) {
                        trsm.fulfill_promise({i,j});
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(potrf.name(j), trsm.name({i,j})));
                        }
                    }
                } else {
                    auto Ljjv = view<double>(blocks[j+j*nb]->data(), n*n);
                    auto isv = view<int>(fulfill[r].data(), fulfill[r].size());                    
                    if(deps_log) {
                        for(auto i: isv) {
                            dlog.add_event(make_unique<DepsEvent>(potrf.name(j), trsm_name({i,j}, r)));
                        }
                    }
                    am_trsm->send(r, Ljjv, j, isv);
                }

            }
        })
        .set_indegree([&](int j) {
            assert(block_2_rank(j,j) == rank);
            if(accumulate_parallel) {
                return j == 0 ? 1 : j; // Need j accumulations into (j,j) to trigger the potf
            } else {
                return 1;
            }
        })
        .set_priority(potf_block_2_prio)
        .set_mapping([&](int j) {
            assert(block_2_rank(j,j) == rank);
            return block_2_thread(j, j);
        })
        .set_name([&](int j) { // This is just for debugging and profiling
            assert(block_2_rank(j,j) == rank);
            return potrf_name(j, rank);
        });

    // Sends a panel (trsm'ed block A(i,j)) and trigger gemms requiring A(i,j)
    auto am_gemm = comm.make_active_msg(
        [&](view<double> &Lij, int& i, int& j, view<int2>& ijs) {
            *blocks[i+j*nb] = Map<MatrixXd>(Lij.data(), n, n);
            for(auto& ij: ijs) {
                gemm.fulfill_promise({j,ij[0],ij[1]});
            }
        });

    /**
     * ij is (Row, Col) of the block in the matrix at A(i,j)
     **/
    trsm.set_task([&](int2 ij) { // A[j,j] & A[i,j] -> A[i,j]
            int i=ij[0]; 
            int j=ij[1]; 
            assert(block_2_rank(i,j) == rank);
            assert(i > j);
            timer t_ = wctime();
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, n, 1.0, blocks[j+j*nb]->data(),n, blocks[i+j*nb]->data(), n);
            timer t__ = wctime();
            trsm_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int2 ij) {
            int i=ij[0];
            int j=ij[1];
            assert(block_2_rank(i,j) == rank);
            assert(i > j);
            vector<vector<int2>> fulfill(n_ranks);
            for (int k = j+1; k < nb; k++) {
                int ii = std::max(i,k);
                int jj = std::min(i,k);
                fulfill[block_2_rank(ii,jj)].push_back({ii,jj});
            }
            for (int r = 0; r < n_ranks; r++)   // Looping through all outgoing dependency edges
            {
                if (r == rank) {
                    for (auto& ij_gemm : fulfill[r]) {
                        gemm.fulfill_promise({j,ij_gemm[0],ij_gemm[1]});
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(trsm.name(ij), gemm.name({j,ij_gemm[0],ij_gemm[1]})));
                        }
                    }
                }
                else {
                    auto Lijv = view<double>(blocks[i+j*nb]->data(), n*n);
                    auto ijsv = view<int2>(fulfill[r].data(), fulfill[r].size());
                    for(auto ij_gemm: ijsv) {
                        if(deps_log) {
                            if(deps_log) {
                                dlog.add_event(make_unique<DepsEvent>(trsm.name(ij), gemm_name({j,ij_gemm[0],ij_gemm[1]}, r)));
                            }
                        }
                    }
                    am_gemm->send(r, Lijv, i, j, ijsv);
                }
            }
        })
        .set_indegree([&](int2 ij) {
            assert(block_2_rank(ij[0],ij[1]) == rank);
            if(accumulate_parallel) {
                return 1 + ij[1]; // Potrf above and all gemms before
            } else {
                return 1 + (ij[1] == 0 ? 0 : 1); // Potrf and last gemm before
            }
        })
        .set_priority(trsm_block_2_prio)
        .set_mapping([&](int2 ij) {
            assert(block_2_rank(ij[0],ij[1]) == rank);
            return block_2_thread(ij[0], ij[1]);
        })
        .set_name([&](int2 ij) { // This is just for debugging and profiling
            assert(block_2_rank(ij[0],ij[1]) == rank);
            return trsm_name(ij, rank);
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    gemm.set_task([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            const int k=kij[0];
            const int i=kij[1];
            const int j=kij[2];
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<MatrixXd> Atmp;
            MatrixXd* Aij;
            double beta = 1.0;
            if(accumulate_parallel) {
                beta = 0.0;
                Atmp = make_unique<MatrixXd>(n, n); // The matrix is allocated with garbage. The 0 in the BLAS call make sure its overwritten by 0's before doing any math
                Aij = Atmp.get();
            } else {
                beta = 1.0;
                Aij = blocks[i+j*nb].get();
            }
            timer t_ = wctime();
            if (i == j) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, -1.0, blocks[i+k*nb]->data(), n, beta, Aij->data(), n);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0,blocks[i+k*nb]->data(), n, blocks[j+k*nb]->data(), n, beta, Aij->data(), n);
            }
            timer t__ = wctime();
            gemm_us_t += 1e6 * elapsed(t_, t__);
            if(accumulate_parallel) {
                lock_guard<mutex> lock(gemm_results[i+j*nb].mtx);
                gemm_results[i+j*nb].to_accumulate[k] = move(Atmp);
            }
        })
        .set_fulfill([&](int3 kij) {
            const int k=kij[0];
            const int i=kij[1];
            const int j=kij[2];
            assert(block_2_rank(kij[1],kij[2]) == rank);
            if(accumulate_parallel) {
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), accu.name(kij)));
                }
                accu.fulfill_promise(kij);
            } else {
                if (k < j-1) {
                    gemm.fulfill_promise({k+1, i, j});
                    if(deps_log) {
                        dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), gemm.name({k+1, i, j})));
                    }
                } else {
                    if (i == j) {
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), potrf.name(i)));
                        }
                        potrf.fulfill_promise(i);
                    } else {
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), trsm.name({i,j})));
                        }
                        trsm.fulfill_promise({i,j});
                    }
                }
            }
        })
        .set_indegree([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            if(accumulate_parallel) {
                return kij[1] == kij[2] ? 1 : 2; // Either one potf or two trsms
            } else {
                return (kij[1] == kij[2] ? 1 : 2) + (kij[0] == 0 ? 0 : 1); // one potrf or two trsms + the gemm before
            }
        })
        .set_priority(gemm_block_2_prio)
        .set_mapping([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return block_2_thread(kij[1], kij[2]); // IMPORTANT if accumulate_parallel is true
        })
        .set_binding([&](int3 kij) {
            return ! accumulate_parallel;
        }).set_name([&](int3 kij) { // This is just for debugging and profiling
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return gemm_name(kij, rank);
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    accu.set_task([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            int k=kij[0]; // Step (gemm's pivot)
            int i=kij[1]; // Row
            int j=kij[2]; // Col
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<Eigen::MatrixXd> Atmp;
            {
                lock_guard<mutex> lock(gemm_results[i+j*nb].mtx);
                Atmp = move(gemm_results[i+j*nb].to_accumulate[k]);
                gemm_results[i+j*nb].to_accumulate.erase(k);
            }
            timer t_ = wctime();
            *blocks[i+j*nb] += (*Atmp);
            timer t__ = wctime();
            accu_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            assert(k < j);
            if(i == j) {
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(accu.name(kij), potrf.name(i)));
                }
                potrf.fulfill_promise(i);
            } else {
                trsm.fulfill_promise({i,j});
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(accu.name(kij), trsm.name({i,j})));
                }
            }
        })
        .set_indegree([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return 1;
        })
        .set_mapping([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return block_2_thread(kij[1], kij[2]); // IMPORTANT. Every (i,j) should map to a given fixed thread
        })
        .set_priority(gemm_block_2_prio)
        .set_binding([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return true; // IMPORTANT
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return accu_name(kij, rank);
        });

    printf("Starting Cholesky factorization...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    timer t0 = wctime();
    if (rank == 0){
        potrf.fulfill_promise(0);
    }
    tp.join();
    MPI_Barrier(MPI_COMM_WORLD);
    timer t1 = wctime();
    printf("Done with Cholesky factorization...\n");
    printf("Elapsed time: %e\n", elapsed(t0, t1));
    printf("Potrf time: %e\n", potrf_us_t.load() * 1e-6);
    printf("Trsm time: %e\n", trsm_us_t.load() * 1e-6);
    printf("Gemm time: %e\n", gemm_us_t.load() * 1e-6);
    printf("Accu time: %e\n", accu_us_t.load() * 1e-6);

    if(log) {
        std::ofstream logfile;
        string filename = "ttor_dist_"+to_string(n)+"_"+to_string(nb)+"_"+ to_string(n_ranks)+"_"+to_string(n_threads)+"_"+to_string(prio_kind)+".log."+to_string(rank);
        logfile.open(filename);
        logfile << logger;
        logfile.close();
    }

    if(deps_log) {
        std::ofstream depsfile;
        string depsfilename = "deps_ttor_dist_"+to_string(n)+"_"+to_string(nb)+"_"+ to_string(n_ranks)+"_"+to_string(n_threads)+"_"+to_string(prio_kind)+".dot."+to_string(rank);
        depsfile.open(depsfilename);
        depsfile << dlog;
        depsfile.close();
    }

    if(test) {
        printf("Starting sending matrix to rank 0...\n");
        // Send the matrix to rank 0
        for (int ii=0; ii<nb; ii++) {
            for (int jj=0; jj<nb; jj++) {
                if (jj<=ii)  {
                    int owner = block_2_rank(ii,jj);
                    MPI_Status status;
                    if (rank == 0 && rank != owner) { // Careful with deadlocks here
                        MPI_Recv(blocks[ii+jj*nb]->data(), n*n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &status);
                    } else if (rank != 0 && rank == owner) {
                        MPI_Send(blocks[ii+jj*nb]->data(), n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        // Rank 0 test
        if(rank == 0) {
            printf("Starting test on rank 0...\n");
            for (int ii=0; ii<nb; ii++) {
                for (int jj=0; jj<nb; jj++) {
                    L.block(ii*n,jj*n,n,n)=*blocks[ii+jj*nb];
                }
            }
            auto L1=L.triangularView<Lower>();
            VectorXd x = VectorXd::Random(n * nb);
            VectorXd b = A*x;
            VectorXd bref = b;
            L1.solveInPlace(b);
            L1.transpose().solveInPlace(b);
            double error = (b - x).norm() / x.norm();
            cout << "Error solve: " << error << endl;
            if(error > 1e-10) {
                printf("\n\nERROR: error is too large!\n\n");
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    int n_threads = 2;
    int verb = 0; // Can be changed to vary the verbosity of the messages
    int n = 5;
    int nb = 10;
    int nprows = 1;
    int npcols = ttor::comm_size();
    PrioKind kind = PrioKind::no;
    bool log = false;
    bool depslog = false;
    bool test = true;
    bool accumulate = false;

    if (argc >= 2)
    {
        n = atoi(argv[1]);
        assert(n > 0);
    }

    if (argc >= 3)
    {
        nb = atoi(argv[2]);
        assert(nb > 0);
    }
    
    if (argc >= 4) {
        n_threads = atoi(argv[3]);
        assert(n_threads > 0);
    }
    
    if (argc >= 5) {
        verb = atoi(argv[4]);
        assert(verb >= 0);
    }

    if (argc >= 6) {
        nprows = atoi(argv[5]);
        assert(nprows >= 0);
    }

    if (argc >= 7) {
        npcols = atoi(argv[6]);
        assert(npcols >= 0);
    }

    if (argc >= 8) {
        assert(atoi(argv[7]) >= 0 && atoi(argv[7]) < 4);
        kind = (PrioKind)atoi(argv[7]);
    }

    if (argc >= 9) {
        log = static_cast<bool>(atoi(argv[8]));
    }

    if (argc >= 10) {
        depslog = static_cast<bool>(atoi(argv[9]));
    }

    if (argc >= 11) {
        test = static_cast<bool>(atoi(argv[10]));
    }

    if (argc >= 12) {
        accumulate = static_cast<bool>(atoi(argv[11]));
    }

    printf("Usage: ./cholesky n nb n_threads verb nprows npcols kind log depslog test accumulate\n");
    printf("Arguments: n (size of blocks) %d\nnb (# of blocks) %d\nn_threads %d\nverb %d\nnprows %d\nnpcols %d\nkind %d\nlog %d\ndeplog %d\ntest %d\naccumulate %d\n", n, nb, n_threads, verb, nprows, npcols, (int)kind, log, depslog, test, accumulate);

    cholesky(n_threads, verb, n, nb, nprows, npcols, kind, log, depslog, test, accumulate);

    MPI_Finalize();
}
