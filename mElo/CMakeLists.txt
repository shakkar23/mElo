# CMakeList.txt : CMake project for mElo, include source and define
# project specific logic here.
#


# Add source to this project's executable.
add_executable (mElo "main.cpp" "main.h"  "mElo/construct_omega.cpp"  "mElo/predict.cpp" "mElo/update_c.cpp")
set_property(TARGET mElo PROPERTY CXX_STANDARD 23)


# TODO: Add tests and install targets if needed.
