#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

#ifndef _OPENMP
#error "Kompilasi dengan flag -fopenmp."
#endif

#define JUMLAH_ULANG 5
#define JUMLAH_KONFIG 5
#define SEED UINT32_C(123456789)
#define NOINLINE __attribute__((noinline))

typedef struct {
    int32_t minimum;
    int32_t maksimum;
    uint64_t jumlah;
    double rata_rata;
    int thread_aktual;
} Hasil;

/* Membaca counter Windows tanpa mengubahnya ke double dahulu. */
static LONGLONG baca_counter(void)
{
    LARGE_INTEGER counter;

    if (!QueryPerformanceCounter(&counter)) {
        fprintf(stderr, "ERROR: pembacaan QPC gagal.\n");
        exit(EXIT_FAILURE);
    }

    return counter.QuadPart;
}

static uint32_t acak_berikutnya(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

static int32_t *buat_dataset(size_t n)
{
    if (n == 0 || n > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "ERROR: ukuran dataset tidak valid.\n");
        return NULL;
    }

    int32_t *data = malloc(n * sizeof(*data));

    if (data == NULL) {
        fprintf(stderr,
                "ERROR: alokasi untuk %zu elemen gagal.\n", n);
        return NULL;
    }

    uint32_t state = SEED;

    for (size_t i = 0; i < n; ++i) {
        data[i] = (int32_t)(acak_berikutnya(&state) >> 16);
    }

    return data;
}

static NOINLINE Hasil hitung_sekuensial(
    const int32_t *data, size_t n)
{
    int32_t minimum = INT32_MAX;
    int32_t maksimum = INT32_MIN;
    uint64_t jumlah = 0;

    for (size_t i = 0; i < n; ++i) {
        int32_t nilai = data[i];

        if (nilai < minimum) {
            minimum = nilai;
        }

        if (nilai > maksimum) {
            maksimum = nilai;
        }

        jumlah += (uint64_t)nilai;
    }

    Hasil hasil;
    hasil.minimum = minimum;
    hasil.maksimum = maksimum;
    hasil.jumlah = jumlah;
    hasil.rata_rata = (double)jumlah / (double)n;
    hasil.thread_aktual = 1;

    return hasil;
}

static NOINLINE Hasil hitung_paralel(
    const int32_t *data, size_t n)
{
    int32_t minimum = INT32_MAX;
    int32_t maksimum = INT32_MIN;
    uint64_t jumlah = 0;
    int thread_aktual = 0;

    #pragma omp parallel for default(none) \
        shared(data, n, thread_aktual) \
        reduction(min:minimum) reduction(max:maksimum) \
        reduction(+:jumlah) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        /*
         * Hanya satu iterasi menulis thread_aktual.
         * Pembacaan dilakukan setelah barrier akhir loop.
         */
        if (i == 0) {
            thread_aktual = omp_get_num_threads();
        }

        int32_t nilai = data[i];

        if (nilai < minimum) {
            minimum = nilai;
        }

        if (nilai > maksimum) {
            maksimum = nilai;
        }

        jumlah += (uint64_t)nilai;
    }

    Hasil hasil;
    hasil.minimum = minimum;
    hasil.maksimum = maksimum;
    hasil.jumlah = jumlah;
    hasil.rata_rata = (double)jumlah / (double)n;
    hasil.thread_aktual = thread_aktual;

    return hasil;
}

static int periksa_hasil(
    Hasil hasil, Hasil acuan, int worker)
{
    if (hasil.thread_aktual != worker) {
        fprintf(stderr,
                "ERROR THREAD: diminta %d, aktual %d.\n",
                worker, hasil.thread_aktual);
        return 0;
    }

    if (hasil.minimum != acuan.minimum
        || hasil.maksimum != acuan.maksimum
        || hasil.jumlah != acuan.jumlah
        || hasil.rata_rata != acuan.rata_rata) {

        fprintf(stderr,
            "ERROR HASIL: perhitungan berbeda dari acuan.\n");

        fprintf(stderr,
            "Acuan: min=%" PRId32 ", max=%" PRId32
            ", sum=%" PRIu64 ", avg=%.12f\n",
            acuan.minimum, acuan.maksimum,
            acuan.jumlah, acuan.rata_rata);

        fprintf(stderr,
            "Hasil: min=%" PRId32 ", max=%" PRId32
            ", sum=%" PRIu64 ", avg=%.12f\n",
            hasil.minimum, hasil.maksimum,
            hasil.jumlah, hasil.rata_rata);

        return 0;
    }

    return 1;
}

int main(int argc, char *argv[])
{
    const size_t ukuran[] = {
        1000, 10000, 100000,
        1000000, 10000000, 100000000
    };

    const int workers[JUMLAH_KONFIG] = {1, 2, 4, 8, 16};
    const size_t jumlah_dataset =
        sizeof(ukuran) / sizeof(ukuran[0]);

    if (argc != 2) {
        fprintf(stderr,
            "Penggunaan: benchmark.exe lokasi_output.csv\n");
        return EXIT_FAILURE;
    }

    if (sizeof(void *) != 8) {
        fprintf(stderr, "ERROR: gunakan compiler 64-bit.\n");
        return EXIT_FAILURE;
    }

    LARGE_INTEGER frekuensi;

    if (!QueryPerformanceFrequency(&frekuensi)
        || frekuensi.QuadPart <= 0) {
        fprintf(stderr,
                "ERROR: frekuensi QPC tidak tersedia.\n");
        return EXIT_FAILURE;
    }

    omp_set_dynamic(0);

    /* Mengganti isi file jika nama output sudah ada. */
    FILE *csv = fopen(argv[1], "w");

    if (csv == NULL) {
        perror("Gagal membuka CSV");
        return EXIT_FAILURE;
    }

    if (fprintf(csv,
        "dataset_id,dataset_size,mode,workers_requested,"
        "workers_actual,iteration,order_in_iteration,"
        "seed,min,max,sum,average,time_seconds,time_ms,"
        "timer,elapsed_ticks,timer_frequency_hz,valid\n"
    ) < 0) {
        fprintf(stderr, "ERROR: gagal menulis header CSV.\n");
        fclose(csv);
        return EXIT_FAILURE;
    }

    printf("EKSPERIMEN MIN, MAX, DAN RATA-RATA\n");
    printf("Compiler          : %s\n", __VERSION__);
    printf("OpenMP            : %d\n", _OPENMP);
    printf("Prosesor logis    : %d\n", omp_get_num_procs());
    printf("Timer pengukuran  : QueryPerformanceCounter\n");
    printf("Frekuensi QPC     : %" PRId64 " Hz\n",
           (int64_t)frekuensi.QuadPart);
    printf("Periode tick QPC  : %.12f detik\n",
           1.0 / (double)frekuensi.QuadPart);
    printf("Tick timer OpenMP : %.12f detik (diagnostik saja)\n",
           omp_get_wtick());
    printf("Target            : 150 pengukuran tercatat\n");
    printf("Output            : %s\n\n", argv[1]);

    size_t jumlah_baris = 0;

    for (size_t d = 0; d < jumlah_dataset; ++d) {
        size_t n = ukuran[d];

        printf("D%zu: menyiapkan %zu elemen...\n", d + 1, n);
        fflush(stdout);

        int32_t *data = buat_dataset(n);

        if (data == NULL) {
            fclose(csv);
            return EXIT_FAILURE;
        }

        /*
         * Acuan dan pemanasan sekuensial.
         * Pembuatan dataset dan pemanasan tidak diukur.
         */
        Hasil acuan = hitung_sekuensial(data, n);

        for (int k = 1; k < JUMLAH_KONFIG; ++k) {
            omp_set_num_threads(workers[k]);

            Hasil pemanasan = hitung_paralel(data, n);

            if (!periksa_hasil(pemanasan, acuan, workers[k])) {
                fprintf(stderr,
                        "Lokasi: pemanasan D%zu.\n", d + 1);
                free(data);
                fclose(csv);
                return EXIT_FAILURE;
            }
        }

        for (int ulang = 0; ulang < JUMLAH_ULANG; ++ulang) {
            for (int posisi = 0;
                 posisi < JUMLAH_KONFIG;
                 ++posisi) {

                /* Rotasi urutan konfigurasi per pengulangan. */
                int k = (posisi + ulang) % JUMLAH_KONFIG;
                int worker = workers[k];
                const char *mode =
                    worker == 1 ? "sequential" : "parallel";

                Hasil hasil;
                LONGLONG mulai;
                LONGLONG selesai;

                if (worker == 1) {
                    mulai = baca_counter();
                    hasil = hitung_sekuensial(data, n);
                    selesai = baca_counter();
                } else {
                    /* Pengaturan ini berada di luar timer. */
                    omp_set_num_threads(worker);

                    mulai = baca_counter();
                    hasil = hitung_paralel(data, n);
                    selesai = baca_counter();
                }

                /*
                 * Selisihkan counter sebagai integer dahulu,
                 * kemudian konversi selisihnya menjadi detik.
                 */
                LONGLONG selisih_tick = selesai - mulai;
                double detik =
                    (double)selisih_tick
                    / (double)frekuensi.QuadPart;

                if (!periksa_hasil(hasil, acuan, worker)) {
                    fprintf(stderr,
                        "Lokasi: D%zu, worker %d, ulangan %d.\n",
                        d + 1, worker, ulang + 1);
                    free(data);
                    fclose(csv);
                    return EXIT_FAILURE;
                }

                if (selisih_tick <= 0) {
                    fprintf(stderr,
                        "ERROR TIMER: durasi tidak positif.\n"
                        "D%zu, worker %d, ulangan %d, "
                        "selisih tick=%" PRId64 "\n",
                        d + 1, worker, ulang + 1,
                        (int64_t)selisih_tick);
                    free(data);
                    fclose(csv);
                    return EXIT_FAILURE;
                }

                /* Seluruh validasi dan I/O di luar timer. */
                int tertulis = fprintf(csv,
                    "D%zu,%zu,%s,%d,%d,%d,%d,%" PRIu32 ","
                    "%" PRId32 ",%" PRId32 ",%" PRIu64 ","
                    "%.12f,%.12f,%.9f,QPC,"
                    "%" PRId64 ",%" PRId64 ",1\n",
                    d + 1,
                    n,
                    mode,
                    worker,
                    hasil.thread_aktual,
                    ulang + 1,
                    posisi + 1,
                    SEED,
                    hasil.minimum,
                    hasil.maksimum,
                    hasil.jumlah,
                    hasil.rata_rata,
                    detik,
                    detik * 1000.0,
                    (int64_t)selisih_tick,
                    (int64_t)frekuensi.QuadPart);

                if (tertulis < 0) {
                    fprintf(stderr,
                            "ERROR: penulisan CSV gagal.\n");
                    free(data);
                    fclose(csv);
                    return EXIT_FAILURE;
                }

                ++jumlah_baris;
            }
        }

        free(data);

        if (fflush(csv) != 0) {
            perror("Gagal menyimpan CSV");
            fclose(csv);
            return EXIT_FAILURE;
        }

        printf("D%zu selesai: 25 pengukuran valid "
               "| Total %zu/150\n\n",
               d + 1, jumlah_baris);
        fflush(stdout);
    }

    if (fclose(csv) != 0) {
        perror("Gagal menutup CSV");
        return EXIT_FAILURE;
    }

    printf("SELESAI: %zu baris pengukuran tersimpan.\n",
           jumlah_baris);

    return EXIT_SUCCESS;
}