#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#define ERT_FLOP 2 
#define ERT_TRIALS_MIN 1
#define ERT_WORKING_SET_MIN 1
#define GBUNIT (1024 * 1024 * 1024)

#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)

#define KERNEL1(a,b,c)   ((a) = (a)*(b))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) +c)

void initialize(uint64_t nsize,
                double* __restrict__ A,
                double value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = value;
  }
}

void kernel(uint64_t nsize,
            uint64_t ntrials,
            double* __restrict__ A,
            int* bytes_per_elem,
            int* mem_accesses_per_elem)
{
  *bytes_per_elem        = sizeof(*A);
  *mem_accesses_per_elem = 2;

  double alpha = 0.5;
  uint64_t i, j;
  for (j = 0; j < ntrials; ++j) {
  for (i = 0; i < nsize; ++i) {
      double beta = 0.8;
      //KERNEL2(beta,A[i],alpha);
      A[i] = beta*A[i]+alpha;
    }
    alpha = alpha * (1 - 1e-8);
  }
}
double getTime()
{
		double time;
		time = omp_get_wtime();
		return time;
}

int main(int argc, char *argv[]) {

		int rank = 0;
		int nprocs = 1;
		int nthreads = 1;
		int id = 0;

		uint64_t TSIZE = 1<<30;
        TSIZE *=16;
		uint64_t PSIZE = TSIZE / nprocs;

		int fd;
 		if ( (fd = open("/mnt/daxtest/f", O_RDWR, 0666)) < 0){
  			printf("open file wrong!\n");
  			exit(1);
 		}
		void * start;
		//
		if ((start=mmap(NULL, TSIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,0))== MAP_FAILED)
		{
			printf("mmap error!\n");
			exit(1);
		}
		double * buf = (double *)malloc(TSIZE);//start;
		//double * buf = (double *)start;
		
		if (buf == NULL) {
				fprintf(stderr, "Out of memory!\n");
				return -1;
		}
#pragma omp parallel private(id) num_threads(160)
//#pragma omp parallel private(id) 
		{
				id = omp_get_thread_num();
				nthreads = omp_get_num_threads();

				uint64_t nsize = PSIZE / nthreads;
				nsize = nsize & (~(64-1));
				nsize = nsize / sizeof(double);
				uint64_t nid =  nsize * id;

				// initialize small chunck of buffer within each thread
				initialize(nsize, &buf[nid], 1.0);


				double startTime, endTime;
				uint64_t n,nNew;
				uint64_t t;
				int bytes_per_elem;
				int mem_accesses_per_elem;

				int it = 0;
				n = nsize; t = 1;
				while (it < 20) { // working set - nsize
						
				#pragma omp barrier

								if ((id == 0) && (rank==0)) {
										startTime = getTime();
								}
								// C-code
								kernel(n, t, &buf[nid], &bytes_per_elem, &mem_accesses_per_elem);
								//msync(&buf[nid], n*sizeof(double), MS_SYNC);
				#pragma omp barrier

								if ((id == 0) && (rank == 0)) {
										endTime = getTime();
										double seconds = (double)(endTime - startTime);
										uint64_t working_set_size = n * nthreads * nprocs;
										uint64_t total_bytes = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
										uint64_t total_flops = t * working_set_size * ERT_FLOP;
										// nsize; trials; microseconds; bytes; single thread bandwidth; total bandwidth
										//printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n",
										//       working_set_size * bytes_per_elem,
										//       t,
										//       seconds,
										//       total_bytes,
										//       total_flops);
                                        printf("BW: %15.3lf Total data: %15.3lfG \n",total_bytes*1.0/seconds/1024/1024/1024, total_bytes*1.0/1024/1024/1024);
								} // print
					it++; 
				} // working set - nsize
		} // parallel region
		//free(buf);
		munmap(buf, TSIZE);
		close(fd);
		printf("\n");
		printf("META_DATA\n");
		printf("FLOPS          %d\n", ERT_FLOP);
		printf("OPENMP_THREADS %d\n", nthreads);

		return 0;
}
