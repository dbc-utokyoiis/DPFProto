#pragma once

// CREATE TABLE ORDERS ( ... );
struct Orders {
    int64_t orderkey;       // O_ORDERKEY       INTEGER NOT NULL
    int64_t custkey;        // O_CUSTKEY        INTEGER NOT NULL
    char orderstatus[1];    // O_ORDERSTATUS    CHAR(1) NOT NULL
    int64_t totalprice;     // O_TOTALPRICE     DECIMAL(15,2) NOT NULL
    int32_t orderdate;      // O_ORDERDATE      DATE NOT NULL
    char orderpriority[15]; // O_ORDERPRIORITY  CHAR(15) NOT NULL
    char clerk[15];         // O_CLERK          CHAR(15) NOT NULL
    int64_t shippriority;   // O_SHIPPRIORITY   INTEGER NOT NULL
    char comment[79];       // O_COMMENT        VARCHAR(79) NOT NULL
};
