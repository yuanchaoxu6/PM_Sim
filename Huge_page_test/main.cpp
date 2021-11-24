#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
#define ERT_FLOP 2 
#define ERT_TRIALS_MIN 1
#define ERT_WORKING_SET_MIN 1
#define GBUNIT (1024 * 1024 * 1024)


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
            uint64_t stride,
            int mem_accesses_per_elem)
{
  double alpha = 0.5;
  double beta = 0.8;
  uint64_t i, j;
  j = 1<<28;
  double value;
  for (i = 0; j>0; j--) {
	  //if (j % 1000000 == 0) printf("j %d\n", j);
      //value = A[i];
	  //asm volatile("" : "+m"(value));
	  A[i] = A[i] * beta;
	  //asm volatile("" : "+m"(A[i]));
	  i=(i+stride) % nsize;
    }
}
double getTime()
{
		double time;
		struct timeval current_time;
  		gettimeofday(&current_time, NULL);
		time = current_time.tv_sec*1000000+current_time.tv_usec;
		return time;
}

int main(int argc, char *argv[]) {

		int rank = 0;
		int nprocs = 1;
		int nthreads = 1;
		int id = 0;
		uint64_t TSIZE = 1<<30;
        TSIZE *=16;
		uint64_t PSIZE = TSIZE / 1;
		int fd;
 		if ( (fd = open("/mnt/daxtest/f", O_RDWR, 0666)) < 0){
  			printf("open file wrong!\n");
  			exit(1);
 		}
		void * start;
		if ((start=mmap(NULL, TSIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,0))== MAP_FAILED)
		{
			printf("mmap error!\n");
			exit(1);
		}
		double * buf = (double *)start;
			//double * buf = (double *)malloc(TSIZE);
			
			uint64_t nsize = PSIZE / 1;
			nsize = nsize & (~(64-1));
			nsize = nsize / sizeof(double);

			initialize(nsize, &buf[0], 1.0);
				// initialize small chunck of buffer within each thread
			double startTime, endTime;
			uint64_t n,nNew;
			uint64_t t;
			int bytes_per_elem;
			int mem_accesses_per_elem;

			uint64_t stride  = 1;
			for (n = 128; n <= TSIZE/8; n=n*2) {
				stride = 1;
				//if ((start=mmap(NULL, n*8, PROT_READ|PROT_WRITE, MAP_SHARED, fd,0))== MAP_FAILED)
				//{
				//	printf("mapping error\n");
				//	exit(0);
				//}
				//double * buf = (double *)start;
				//double * buf = (double *)malloc(n*8);
					
				while (stride < n) { // working set - nsize
					uint64_t ntrials;
					//ntrials = sqrt(it);
					
					asm volatile ("mfence" ::: "memory");
					startTime = getTime();
									// C-code
					kernel(n, 0, &buf[0], stride, mem_accesses_per_elem);
									//msync(&buf[nid], n*sizeof(double), MS_SYNC);
					asm volatile ("mfence" ::: "memory");
					endTime = getTime();
					double useconds = (double)(endTime - startTime);
					printf("%12llu %12llu %.3lf\n", n, stride, useconds);
					stride = stride * 2; 
					
				} // working set - nsize
				//free(buf);
				//munmap(start, n*8);
			}
	    munmap(buf, TSIZE);
		//free(buf);
		close(fd);
		printf("\n");
		printf("META_DATA\n");

		return 0;
}
