#include <cstdio>
#include <cstring>
#include <iostream>
#include <time.h>
#include "vv-aes.h"


/* Store each subsituted character in the corresponding element in another 
array based on the element index */

uint8_t character_mapping[UNIQUE_CHARACTERS];

void create_mapping(){

    for (int i = 0; i < UNIQUE_CHARACTERS; i++) {
        character_mapping[originalCharacter[i]] = substitutedCharacter[i];
        }
}

/**
 * This function takes the characters stored in the 4x4 message array and substitutes each character for the
 * corresponding replacement as specified in the originalCharacter and substitutedCharacter array.
 * This corresponds to step 2.1 in the VV-AES explanation.
 */

void substitute_bytes() {
    // For each byte in the message
    for (int column = 0; column < MESSAGE_BLOCK_WIDTH; column++) {
        for (int row = 0; row < MESSAGE_BLOCK_WIDTH; row++) {
            message[row][column] = character_mapping[message[row][column]];
        }
    }
}


/*
 * This function shifts (rotates) a row in the message array by one place to the left.
 * @param row The row which to shift.
 */


/*
 * This function shifts each row by the number of places it is meant to be shifted according to the AES specification.
 * Row zero is shifted by zero places. Row one by one, etc.
 * This corresponds to step 2.2 in the VV-AES explanation.
 */

void shift_rows(){

    unsigned char temp,temp1,temp2;
    temp = message[1][0];
    message[1][0] = message[1][1];
    message[1][1] = message[1][2];
    message[1][2] = message[1][3];
    message[1][3] = temp;

    temp = message[2][0];
    temp1 = message[2][1];
    message[2][0] = message[2][2];
    message[2][1] = message[2][3];
    message[2][2] = temp;
    message[2][3] = temp1;

    temp = message[3][0];
    temp1 = message[3][1];
    temp2 = message[3][2];
    message[3][0] = message[3][3];
    message[3][1] = temp;
    message[3][2] = temp1;
    message[3][3] = temp2;
}

/*
 * This function evaluates four different polynomials, one for each row in the column.
 * Each polynomial evaluated is of the form
 * m'[row, column] = c[r][3] m[3][column]^4 + c[r][2] m[2][column]^3 + c[r][1] m[1][column]^2 + c[r][0]m[0][column]^1
 * where m' is the new message value, c[r] is an array of polynomial coefficients for the current result row (each
 * result row gets a different polynomial), and m is the current message value.
 
 * For each column, mix the values by evaluating them as parameters of multiple polynomials.
 * This corresponds to step 2.3 in the VV-AES explanation.
 */


void mix_columns() {

    for (unsigned int column = 0; column < MESSAGE_BLOCK_WIDTH; ++column) {
        for (unsigned int row = 0; row < MESSAGE_BLOCK_WIDTH; ++row) {
        int result = 0;
        
            result += polynomialCoefficients[row][0] * message[0][column];
            result += polynomialCoefficients[row][1] * message[1][column] * message[1][column];
            result += polynomialCoefficients[row][2] * message[2][column] * message[2][column] * message[2][column];
            result += polynomialCoefficients[row][3] * message[3][column] * message[3][column] * message[3][column] * message[3][column];
        message[row][column] = result;
        }
    }
}

/*
 * Add the current key to the message using the XOR operation.
 */
void add_key() {
    for (int column = 0; column < MESSAGE_BLOCK_WIDTH; column++) {
        for (int row = 0; row < MESSAGE_BLOCK_WIDTH; ++row) {
            // ^ == XOR
            message[row][column] = message[row][column] ^ key[row][column];
        }
    }
}


int main() {
    // Receive the problem from the system.
    readInput();

    create_mapping();
    // For extra security (and because Varys wasn't able to find enough test messages to keep you occupied) each message
    // is put through VV-AES lots of times. If we can't stop the adverse Maesters from decrypting our highly secure
    // encryption scheme, we can at least slow them down.
    for (int i = 0; i < 500000; i++) {
        // For each message, we use a predetermined key (e.g. the password). In our case, its just pseudo random.
        set_next_key();

        // First, we add the key to the message once:
        add_key();

        // We do 9+1 rounds for 128 bit keys
        for (int round = 0; round < 9; round++) {
            //In each round, we use a different key derived from the original (refer to the key schedule).
            set_next_key();

            // These are the four steps described in the slides.
            substitute_bytes();
            shift_rows();
            mix_columns();
            add_key();
        }
        // Final round
        substitute_bytes();
        shift_rows();
        add_key();
    }

    // Submit our solution back to the system.
    writeOutput();
    return 0;
}

