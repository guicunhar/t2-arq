/**
 * @file gerador_dados.c
 * @brief Utilitário para gerar arquivos binários de teste (matriz A e vetor B)
 *        para o programa "equation" do Trabalho 2.
 *
 * Gera uma matriz A quadrada n x n diagonalmente dominante (garante pivôs
 * não nulos durante toda a eliminação, mesmo sem pivotamento parcial) e um
 * vetor B de tamanho n, ambos em float (data_t), salvos em formato binário
 * puro, exatamente como leMatriz/leVetor (em equation_test.cu) esperam.
 *
 * Uso:
 *   ./gerador_dados <n> <arquivo_matrizA.bin> <arquivo_vetorB.bin> [semente]
 *
 * Exemplo:
 *   ./gerador_dados 1024 matrizA.bin vetorB.bin 42
 *
 * Compilação:
 *   gcc -Wall -Wextra -O2 -o gerador_dados gerador_dados.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef float data_t;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <n> <arquivo_matrizA.bin> <arquivo_vetorB.bin> [semente]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    const char *nomeA = argv[2];
    const char *nomeB = argv[3];
    unsigned int semente = (argc > 4) ? (unsigned int)atoi(argv[4]) : 1234u;

    if (n <= 0 || n % 8 != 0) {
        fprintf(stderr, "[ERROR] n deve ser positivo e multiplo de 8 (recebido: %d)\n", n);
        return 1;
    }

    srand(semente);

    data_t *A = (data_t *)malloc(sizeof(data_t) * (size_t)n * (size_t)n);
    data_t *B = (data_t *)malloc(sizeof(data_t) * (size_t)n);
    if (!A || !B) {
        fprintf(stderr, "[ERROR] Falha ao alocar memoria para n=%d\n", n);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        data_t somaLinha = 0.0f;
        for (int j = 0; j < n; j++) {
            data_t valor = (data_t)((rand() % 2001) - 1000) / 100.0f; /* [-10, 10] */
            A[i * n + j] = valor;
            somaLinha += fabsf(valor);
        }
        /* diagonalmente dominante: |a_ii| > soma dos demais |a_ij| da linha */
        A[i * n + i] = somaLinha + 50.0f;
        B[i] = (data_t)((rand() % 2001) - 1000) / 10.0f; /* [-100, 100] */
    }

    FILE *arqA = fopen(nomeA, "wb");
    if (!arqA) { fprintf(stderr, "[ERROR] Nao foi possivel criar %s\n", nomeA); return 1; }
    fwrite(A, sizeof(data_t), (size_t)n * (size_t)n, arqA);
    fclose(arqA);

    FILE *arqB = fopen(nomeB, "wb");
    if (!arqB) { fprintf(stderr, "[ERROR] Nao foi possivel criar %s\n", nomeB); return 1; }
    fwrite(B, sizeof(data_t), (size_t)n, arqB);
    fclose(arqB);

    printf("Gerado: %s (%dx%d floats) e %s (%d floats), semente=%u\n",
           nomeA, n, n, nomeB, n, semente);

    free(A);
    free(B);
    return 0;
}