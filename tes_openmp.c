#include <stdio.h>
#include <omp.h>

#ifndef _OPENMP
#error "OpenMP belum aktif. Gunakan flag -fopenmp."
#endif

int main(void)
{
    int jumlah_thread = 0;

    omp_set_dynamic(0);
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        #pragma omp single
        {
            jumlah_thread = omp_get_num_threads();
        }
    }

    printf("OpenMP aktif   : %d\n", _OPENMP);
    printf("Ukuran pointer : %zu byte\n", sizeof(void *));
    printf("Prosesor logis : %d\n", omp_get_num_procs());
    printf("Thread diminta : 4\n");
    printf("Thread aktual  : %d\n", jumlah_thread);

    if (jumlah_thread == 4 && sizeof(void *) == 8) {
        printf("STATUS: LULUS\n");
        return 0;
    }

    printf("STATUS: PERLU DIPERIKSA\n");
    return 1;
}