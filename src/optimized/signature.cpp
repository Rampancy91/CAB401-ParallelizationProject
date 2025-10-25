#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "uthash.h"
#include <chrono>

#include <omp.h>
#include <vector>


typedef unsigned char byte;

#define SIGNATURE_LEN 64

int DENSITY  = 21;
int PARTITION_SIZE;

int inverse[256];
const char* alphabet = "CSTPAGNDEQHRKMILVFYW";


void seed_random(char* term, int length);
short random_num(short max);
void Init();

int doc_sig[SIGNATURE_LEN];

int WORDLEN;
FILE *sig_file;

typedef struct
{
    char term[100];
    short sig[SIGNATURE_LEN];
    UT_hash_handle hh;
} hash_term;

hash_term *vocab = NULL;

//pre-compute signatures array
// Only 20^3 = 8000 possible combinations

#define MAX_KMERS 8000
short precomputed_sigs[MAX_KMERS][SIGNATURE_LEN];
bool use_precomputed = true; //To toggle between methods

//Structure to hold partitions signature data

struct PartitionSignature {
    int doc_id;
    byte signature[SIGNATURE_LEN / 8];
};


//K-mer index functions
//Convert kmer to unique index
int kmer_to_index(char* term)
{
    int index = 0;
    for (int i = 0; i < WORDLEN; i++)
    {
        index = index * 20 + inverse[(unsigned char)term[i]];
    }
    return index;
}

//Convert index back to k-mer
void index_to_kmer(int index, char* term)
{
    for (int i = WORDLEN - 1; i >= 0; i--)
    {
        term[i] = alphabet[index % 20];
        index /= 20;
    }
    term[WORDLEN] = '\0';
}

short* compute_new_term_sig(char* term, short *term_sig)
{
    seed_random(term, WORDLEN);

    int non_zero = SIGNATURE_LEN * DENSITY/100;

    int positive = 0;
    while (positive < non_zero/2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero/2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = -1;
            negative++;
        }
    }
    return term_sig;
}

//Pre compute all 8000 possible signatures
void precompute_all_signatures()
{
    printf("Pre-computing all %d possible %d-mer signatures...\n", MAX_KMERS, WORDLEN);

    auto start = std::chrono::high_resolution_clock::now();

    char term[10];
    for (int i = 0; i < MAX_KMERS; i++)
    {
        // Convert index to k-mer
        index_to_kmer(i, term);

        // Generate signature
        memset(precomputed_sigs[i], 0, sizeof(precomputed_sigs[i]));
        compute_new_term_sig(term, precomputed_sigs[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    printf("Pre-computation complete in %f seconds\n", duration.count());
}

// New version -  signature lookup using pre-computed array
short* find_sig_fast(char* term)
{
    int index = kmer_to_index(term);
    return precomputed_sigs[index];
}

// Old version - hash table look up
short *find_sig_hash(char* term)
{
    hash_term *entry;
    HASH_FIND(hh, vocab, term, WORDLEN, entry);
    if (entry == NULL)
    {
        entry = (hash_term*)malloc(sizeof(hash_term));
        strncpy_s(entry->term, sizeof(entry->term), term, WORDLEN);
        memset(entry->sig, 0, sizeof(entry->sig));
        compute_new_term_sig(term, entry->sig);
        HASH_ADD(hh, vocab, term, WORDLEN, entry);
    }

    return entry->sig;
}

//Use selected method
short* find_sig(char* term)
{
    if (use_precomputed)
        return find_sig_fast(term);
    else
        return find_sig_hash(term);
}


void signature_add(char* term)
{
	short* term_sig = find_sig(term);
	for (int i=0; i<SIGNATURE_LEN; i++)
		doc_sig[i] += term_sig[i];
}

int doc = 0;

void compute_signature(char* sequence, int length)
{
    memset(doc_sig, 0, sizeof(doc_sig));

    for (int i=0; i<length-WORDLEN+1; i++)
        signature_add(sequence+i);

    // save document number to sig file
    fwrite(&doc, sizeof(int), 1, sig_file);
    
    // flatten and output to sig file
    for (int i = 0; i < SIGNATURE_LEN; i += 8) 
    {
        byte c = 0;
        for (int j = 0; j < 8; j++) 
            c |= (doc_sig[i+j]>0) << (7-j);
        fwrite(&c, sizeof(byte), 1, sig_file);
    }
}

#define min(a,b) ((a) < (b) ? (a) : (b))

void partition(char* sequence, int length)
{
    int i=0;
    do
    {
        compute_signature(sequence+i, min(PARTITION_SIZE, length-i));
        i += PARTITION_SIZE/2;
    }
    while (i+PARTITION_SIZE/2 < length);
    doc++;
}

int power(int n, int e)
{
    int p = 1;
    for (int j=0; j<e; j++)
        p *= n;
    return p;
}

int main(int argc, char* argv[])
{
    //const char* filename = "qut2.fasta";
    //const char* filename = "qut3.fasta";

    const char* filename; 

    // Allowing command line input or default to small test file

    if (argc > 1) {
        filename = argv[1];
    }
    else {
        filename = "test.fasta";  // Default
    }
    
    WORDLEN = 3;
    PARTITION_SIZE = 16;
    int WORDS = power(20, WORDLEN);

    for (int i=0; i<strlen(alphabet); i++)
        inverse[alphabet[i]] = i;

    //Pre-compute signatures
    if (use_precomputed) 
    {
        precompute_all_signatures();
    }

    auto start = std::chrono::high_resolution_clock::now();

    FILE* file;
    errno_t OK = fopen_s(&file, filename, "r");

    if (OK != 0)
    {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return 1;
    }

    char outfile[256];
    sprintf_s(outfile, 256, "%s.part%d_sigs%02d_%d", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);
    fopen_s(&sig_file, outfile, "w");

    char buffer[10000];
    while (!feof(file))
    {
        fgets(buffer, 10000, file); // skip meta data line
        fgets(buffer, 10000, file);
        int n = (int)strlen(buffer) - 1;
        buffer[n] = 0;
        partition(buffer, n);
    }
    fclose(file);

    fclose(sig_file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    printf("%s %f seconds\n", filename, duration.count());

    // Show which method was used
    if (use_precomputed)
    {
        printf("Method: Pre-computed array lookup\n");
    }
    else
    {
        printf("Method: Hash table lookup\n");
    }

    return 0;
}
