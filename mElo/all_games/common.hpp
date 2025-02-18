#pragma once


// common game struct for interfacing with the trainer throughout all games
struct game {
	// indices into the matrix
	int player1;
	int player2;
	// 0 or 1
	float outcome;
};



// the inline keyword is used to make sure that the variable is only defined once throughout all translation units
// 
// this maps the name to the index in the matrix
inline std::unordered_map<std::string, int> player_map;


