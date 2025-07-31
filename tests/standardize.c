#include <stdio.h>
#include <math.h>   // sqrt()関数を使うために必要
#include <stdlib.h> // malloc()とfree()関数を使うために必要

// 関数のプロトタイプ宣言
void standardize(const double* x, double* y, int size);

#define DDOF 0

int main() {
    // 1. テストデータの準備
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int size = sizeof(x) / sizeof(x[0]);

    // 2. 出力用の配列を動的に確保
    double* y = (double*)malloc(size * sizeof(double));
    if (y == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 3. 標準化関数を呼び出す
    standardize(x, y, size);

    // 4. 結果を表示
    printf("Original data (x):\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", x[i]);
    }
    printf("\n\n");

    printf("Standardized data (y):\n");
    for (int i = 0; i < size; i++) {
        // 結果が-1.5 ~ 1.5の範囲に収まることが多い
        // Pythonのnumpy.std([1,2,3,4,5])は1.414...
        // y[0] = (1-3)/1.414 = -1.414...
        printf("% .4f ", y[i]); 
    }
    printf("\n");

    // 5. 確保したメモリを解放
    free(y);

    return 0;
}

/**
 * @brief 配列を標準化する関数 y = (x - mean(x)) / (std(x) + epsilon)
 * @param x 入力配列 (constなので関数内で変更されない)
 * @param y 出力配列 (計算結果が格納される)
 * @param size 配列の要素数
 */
void standardize(const double* x, double* y, int size) {
    // エッジケース: 要素数が1以下の場合は標準化できない
    if (size <= 1) {
        if (size == 1) {
            y[0] = 0.0; // 偏差が0なので結果は0
        }
        return; // 要素数0の場合は何もしない
    }

    // --- ステップ1: 平均 (mean) の計算 ---
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += x[i];
    }
    double mean = sum / size;

    // --- ステップ2: 標準偏差 (std) の計算 ---
    // (numpy.stdのデフォルトである母標準偏差を計算)
    double sum_sq_diff = 0.0;
    for (int i = 0; i < size; i++) {
        // (x[i] - mean) の2乗を足していく
        sum_sq_diff += (x[i] - mean) * (x[i] - mean);
    }

#if !DDOF
    // 母分散 (Nで割る)
    double variance = sum_sq_diff / size;
#else
    // 不偏分散 (N-1で割る)
    double variance = sum_sq_diff / (size - 1);
#endif

    // 標準偏差
    double std_dev = sqrt(variance);

    // --- ステップ3: 標準化の計算 ---
    // ゼロ除算を避けるための小さな値 (イプシロン)
    const double epsilon = 1e-8;
    double denominator = std_dev + epsilon;

    for (int i = 0; i < size; i++) {
        y[i] = (x[i] - mean) / denominator;
    }
}

// Original data (x):
// 1.00 2.00 3.00 4.00 5.00 
// 
// Standardized data (y):
// -1.4142 -0.7071  0.0000  0.7071  1.4142 

// (x - x.mean()) / x.std() 


// 補足: 標本標準偏差を使いたい場合
// 統計的な文脈で、手元のデータが「標本」であり母集団を推定したい場合は、標本標準偏差（不偏標準偏差）を使います。その場合、分散を計算するときに N の代わりに N-1 で割ります。
// Cコードを修正するには、standardize 関数内の以下の行を変更します。
// 
// // 変更前 (母分散)
// double variance = sum_sq_diff / size;
// 
// // 変更後 (不偏分散)
// double variance = sum_sq_diff / (size - 1);
