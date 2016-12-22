# KOKKOS_PATH (root directory of the kokkos repo) must be passed in
#KOKKOS_PATH =
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "kokkos_basic"


SRC = $(wildcard *.cpp)
#vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = ""
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =  

DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

