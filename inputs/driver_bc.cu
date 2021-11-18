#include <string>
void BC (char* filename, int, float t);
int main(int argc, char* argv[]) {
	BC(argv[1], atoi(argv[2]), atof(argv[3]));
}
