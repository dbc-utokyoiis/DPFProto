#pragma once

#define DBG_PERROR(s)       perror(s)
#define DBG_ERROR(fmt, ...) fprintf(stderr, "" fmt "\n", ## __VA_ARGS__)

#define PANIC_PERROR(...)       \
  {             \
    DBG_PERROR("" __VA_ARGS__); \
    exit(EXIT_FAILURE);         \
  }


#define PANIC_ERROR(...)       \
  {             \
    DBG_PERROR("" __VA_ARGS__); \
    exit(EXIT_FAILURE);         \
  }

