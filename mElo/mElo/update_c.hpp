#pragma once

#include "Number.hpp"
#include "types.hpp"

std::tuple<Numeric, Numeric, Column, Column> update_melo(Numeric rA, Numeric rB, Column cA, Column cB, Numeric observed_score);
