#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

/* Generator pseudoacak dengan state 32-bit.
   Seed awal harus bukan nol. */
static uint32_t acak_berikutnya(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

/* Membuat array di heap dan mengisinya secara deterministik. */
static int32_t *buat_dataset(size_t n, uint32_t seed)
{
    if (n == 0 || seed == 0) {
        fprintf(stderr, "ERROR: ukuran atau seed tidak valid.\n");
        return NULL;
    }

    /* Cegah overflow saat menghitung kebutuhan memori. */
    if (n > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "ERROR: ukuran dataset terlalu besar.\n");
        return NULL;
    }

    int32_t *data = malloc(n * sizeof(*data));

    if (data == NULL) {
        fprintf(stderr,
                "ERROR: alokasi memori untuk %zu elemen gagal.\n",
                n);
        return NULL;
    }

    uint32_t state = seed;

    for (size_t i = 0; i < n; ++i) {
        data[i] = (int32_t)(acak_berikutnya(&state) >> 16);
    }

    return data;
}

int main(void)
{
    const size_t ukuran[] = {
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000
    };

    const size_t jumlah_ukuran = sizeof(ukuran) / sizeof(ukuran[0]);
    const uint32_t seed = UINT32_C(123456789);

    printf("UJI GENERATOR DATASET\n");
    printf("Seed          : %" PRIu32 "\n", seed);
    printf("Rentang nilai : 0 sampai 65535\n");
    printf("Ukuran elemen : %zu byte\n\n", sizeof(int32_t));

    for (size_t d = 0; d < jumlah_ukuran; ++d) {
        size_t n = ukuran[d];
        double memori_mib =
            (double)(n * sizeof(int32_t)) / (1024.0 * 1024.0);

        printf("D%zu: %zu elemen | %.2f MiB\n",
               d + 1, n, memori_mib);
        fflush(stdout);

        int32_t *data = buat_dataset(n, seed);

        if (data == NULL) {
            return EXIT_FAILURE;
        }

        printf("  Lima nilai pertama:");

        for (size_t i = 0; i < 5; ++i) {
            printf(" %" PRId32, data[i]);
        }

        printf("\n");

        /* Total pembanding untuk memeriksa reproduksibilitas.
           uint64_t cukup untuk seluruh ukuran pada pengujian ini. */
        uint64_t total = 0;

        for (size_t i = 0; i < n; ++i) {
            total += (uint64_t)data[i];
        }

        printf("  Total pembanding: %" PRIu64 "\n", total);

        free(data);
        data = NULL;

        printf("  STATUS: BERHASIL, memori dibebaskan.\n\n");
    }

    printf("SELESAI: seluruh dataset berhasil dibuat dan diperiksa.\n");

    return EXIT_SUCCESS;
}