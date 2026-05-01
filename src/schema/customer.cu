#pragma once

// CREATE TABLE CUSTOMER ( ... );
struct Customer {
    int64_t custkey;     // C_CUSTKEY     INTEGER NOT NULL
    char name[25];       // C_NAME        VARCHAR(25) NOT NULL
    char address[40];    // C_ADDRESS     VARCHAR(40) NOT NULL
    int64_t nationkey;   // C_NATIONKEY   INTEGER NOT NULL
    char phone[15];      // C_PHONE       CHAR(15) NOT NULL
    int64_t acctbal;     // C_ACCTBAL     DECIMAL(15,2) NOT NULL
    char mktsegment[10]; // C_MKTSEGMENT  CHAR(10) NOT NULL
    char comment[117];   // C_COMMENT     VARCHAR(117) NOT NULL
};
