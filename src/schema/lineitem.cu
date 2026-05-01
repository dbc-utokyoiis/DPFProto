#pragma once

// CREATE TABLE LINEITEM ( ... );
struct Lineitem {
    int64_t orderkey;      // L_ORDERKEY       INTEGER NOT NULL
    int64_t partkey;       // L_PARTKEY        INTEGER NOT NULL
    int64_t suppkey;       // L_SUPPKEY        INTEGER NOT NULL
    int64_t linenumber;    // L_LINENUMBER     INTEGER NOT NULL
    int64_t quantity;      // L_QUANTITY       DECIMAL(15,2) NOT NULL
    int64_t extendedprice; // L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL
    int64_t discount;      // L_DISCOUNT       DECIMAL(15,2) NOT NULL
    int64_t tax;           // L_TAX            DECIMAL(15,2) NOT NULL
    char returnflag[1];    // L_RETURNFLAG     CHAR(1) NOT NULL
    char linestatus[1];    // L_LINESTATUS     CHAR(1) NOT NULL
    int32_t shipdate;      // L_SHIPDATE       DATE NOT NULL
    int32_t commitdate;    // L_COMMITDATE     DATE NOT NULL
    int32_t receiptdate;   // L_RECEIPTDATE    DATE NOT NULL
    char shipinstruct[25]; // L_SHIPINSTRUCT   CHAR(25) NOT NULL
    char shipmode[10];     // L_SHIPMODE       CHAR(10) NOT NULL
    char comment[44];      // L_COMMENT        VARCHAR(44) NOT NULL
};
