#include <stdio.h>
#include <stdlib.h>

/*
Para compilar:
gcc -o test/file_gen test/file_gen.c

Para executar:
./test/file_gen N_LINHAS N_COLUNAS NOME_DO_ARQUIVO.dat
Exemplo:
./test/file_gen 3 4 test/matrix_3x4.dat
./test/file_gen 4 3 test/matrix_4x3.dat
*/


int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <linhas> <colunas> <arquivo_saida>\n", argv[0]);
        return 1;
    }

    unsigned long int linhas = strtoul(argv[1], NULL, 10);
    unsigned long int colunas = strtoul(argv[2], NULL, 10);
    char *filename = argv[3];

    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Erro ao abrir arquivo %s\n", filename);
        return 1;
    }

    for (unsigned long int i = 0; i < linhas * colunas; i++) {
        float value = (float)(i + 1);  // pode ser sequencial
        // float value = (float)rand() / RAND_MAX; // ou aleatório entre 0 e 1
        fwrite(&value, sizeof(float), 1, f);
    }

    fclose(f);
    printf("Arquivo %s gerado com %lu floats.\n", filename, linhas * colunas);
    return 0;
}
