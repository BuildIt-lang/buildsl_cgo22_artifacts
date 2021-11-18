#include <string>
void BFS (char* filename, int, float t);
int main(int argc, char* argv[]) {
	BFS(argv[1], atoi(argv[2]), atof(argv[3]));
}
