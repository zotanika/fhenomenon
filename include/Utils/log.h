#pragma once

#include <iostream>

#define FHENOMENON_DEBUG

#ifdef FHENOMENON_DEBUG
#define LOG_MESSAGE(msg) std::cout << msg << std::endl;
#else
#define LOG_MESSAGE(msg) // do nothing
#endif
