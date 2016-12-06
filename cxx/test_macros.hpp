#ifndef TEST_MACROS_HPP
#define TEST_MACROS_HPP

typedef double    real;

#define NUM_POINTS       4
#define NUM_CORNERS      4
#define NUM_ELEMENTS    10
#define NUM_LEVELS       5
#define NUM_LEVELS_P     NUM_LEVELS+1
#define NUM_TIME_LEVELS  3
#define QSIZE_D          4

#define ACCESS_IP_JP_IE (pointer, ip, jp, ie) \
   pointer[ NUM_ELEMENTS*NUM_POINTS*ip + NUM_ELEMENTS*jp + ie ]

#endif // TEST_MACROS_HPP
