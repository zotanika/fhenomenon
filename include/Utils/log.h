#pragma once

#include <iostream>

#if defined(FHENOMENON_DEBUG) && FHENOMENON_DEBUG
#define LOG_MESSAGE(msg)                                                                                               \
  do {                                                                                                                 \
    std::cout << msg << std::endl;                                                                                     \
  } while (false)
#else
#define LOG_MESSAGE(msg)                                                                                               \
  do {                                                                                                                 \
  } while (false)
#endif
