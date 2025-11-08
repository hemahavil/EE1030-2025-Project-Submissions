#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

double* createMatrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    return mat;
}

void transposeMatrix(double* in, double* out, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j * rows + i] = in[i * cols + j];
}

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++)
                C[i * p + j] += A[i * n + k] * B[k * p + j];
        }
}

void matVecMultiply(double* A, double* x, double* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0;
        for (int j = 0; j < cols; j++)
            y[i] += A[i * cols + j] * x[j];
    }
}

// power iteration method - finds eigenvalues and eigenvectors
void powerIteration(double* M, int n, double* eigvals, double* eigvecs, int k) {
    double* W = createMatrix(n, n);
    for (int i = 0; i < n * n; i++) W[i] = M[i];

    double* b = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));

    for (int comp = 0; comp < k; comp++) {
        // initialize with random values
        for (int i = 0; i < n; i++) b[i] = (double)rand() / RAND_MAX;

        double lambda = 0, newLambda;
        for (int iter = 0; iter < 800; iter++) {
            matVecMultiply(W, b, y, n, n);
            
            // normalize the vector
            double norm = 0;
            for (int i = 0; i < n; i++) norm += y[i] * y[i];
            norm = sqrt(norm);
            for (int i = 0; i < n; i++) b[i] = y[i] / norm;

            matVecMultiply(W, b, y, n, n);
            newLambda = 0;
            for (int i = 0; i < n; i++) newLambda += b[i] * y[i];

            if (fabs(newLambda - lambda) < 1e-6) break;
            lambda = newLambda;
        }

        eigvals[comp] = lambda;
        for (int i = 0; i < n; i++) eigvecs[comp * n + i] = b[i];

        // deflate matrix to find next eigenvalue
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                W[i * n + j] -= lambda * b[i] * b[j];
    }
    free(W); free(b); free(y);
}

// compute SVD using A^T * A method
void computeSVD(double* A, int rows, int cols, int k, double* S, double* U, double* V) {
    double* At = createMatrix(cols, rows);
    double* AtA = createMatrix(cols, cols);
    
    transposeMatrix(A, At, rows, cols);
    matrixMultiply(At, A, AtA, cols, rows, cols);

    double* eigvals = (double*)malloc(k * sizeof(double));
    powerIteration(AtA, cols, eigvals, V, k);

    // singular values = sqrt(eigenvalues)
    for (int i = 0; i < k; i++) S[i] = sqrt(fabs(eigvals[i]));

    double* v = (double*)malloc(cols * sizeof(double));
    double* u = (double*)malloc(rows * sizeof(double));

    // compute U vectors
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < cols; j++) v[j] = V[i * cols + j];
        matVecMultiply(A, v, u, rows, cols);
        for (int r = 0; r < rows; r++)
            U[i * rows + r] = (S[i] > 1e-8) ? u[r] / S[i] : 0;
    }

    free(At); free(AtA); free(eigvals); free(v); free(u);
}

void reconstructImage(double* U, double* S, double* V, int rows, int cols, int k, double* out) {
    for (int i = 0; i < rows * cols; i++) out[i] = 0;
    
    // reconstruct using rank-k approximation
    for (int i = 0; i < k; i++)
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                out[r * cols + c] += S[i] * U[i * rows + r] * V[i * cols + c];
}

void imageToMatrix(unsigned char* img, int w, int h, int ch, double* mat) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int idx = (i * w + j) * ch;
            // convert to grayscale if needed
            mat[i * w + j] = (ch == 1) ? img[idx] : 
                0.299 * img[idx] + 0.587 * img[idx+1] + 0.114 * img[idx+2];
        }
}

void matrixToImage(double* mat, int w, int h, unsigned char* img) {
    for (int i = 0; i < h * w; i++) {
        double val = mat[i];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        img[i] = (unsigned char)val;
    }
}

// calculate error between original and compressed
double calcError(double* orig, double* compressed, int size) {
    double err = 0;
    for (int i = 0; i < size; i++) {
        double diff = orig[i] - compressed[i];
        err += diff * diff;
    }
    return sqrt(err);
}

int main() {
    const char* images[] = {
        "../images/einstein.jpg",
        "../images/grayscale.jpg",
        "../images/globe.jpg"
    };
    
    int k = 40;  // number of singular values to keep
    // int k = 20;  // tried with 20 first but quality was bad

    printf("Starting image compression with k=%d\n\n", k);

    for (int idx = 0; idx < 3; idx++) {
        printf("Processing image %d: %s\n", idx+1, images[idx]);

        int w, h, ch;
        unsigned char* img = stbi_load(images[idx], &w, &h, &ch, 0);
        if (!img) {
            printf("Error loading image!\n");
            continue;
        }
        
        //printf("Image size: %dx%d, channels: %d\n", w, h, ch);

        double* A = createMatrix(h, w);
        imageToMatrix(img, w, h, ch, A);

        double* S = (double*)malloc(k * sizeof(double));
        double* U = createMatrix(k, h);
        double* V = createMatrix(k, w);
        
        printf("  Computing SVD...\n");
        computeSVD(A, h, w, k, S, U, V);

        double* recon = createMatrix(h, w);
        reconstructImage(U, S, V, h, w, k, recon);

        // calculate compression error for report
        double err = calcError(A, recon, h * w);
        
        // need to calculate relative error
        double orig_norm = 0;
        for(int i = 0; i < h*w; i++) {
            orig_norm += A[i] * A[i];
        }
        orig_norm = sqrt(orig_norm);
        double rel_err = (err / orig_norm) * 100.0;
        
        printf("  Error: %.2f (%.2f%% relative)\n", err, rel_err);

        unsigned char* outImg = (unsigned char*)malloc(w * h);
        matrixToImage(recon, w, h, outImg);

        char name[100];
        sprintf(name, "../images/compressed_%d.jpg", idx + 1);
        stbi_write_jpg(name, w, h, 1, outImg, 90);
        printf("  Saved to: %s\n\n", name);

        free(S); free(U); free(V); free(recon); free(outImg);
        stbi_image_free(img); 
        free(A);
    }

    printf("All done!\n");
    return 0;
}