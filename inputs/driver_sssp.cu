#include <string>
void SSSP (char* filename, int, int t);
int main(int argc, char* argv[]) {
	SSSP(argv[1], atoi(argv[2]), atoi(argv[3]));
}
