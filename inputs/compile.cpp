#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
using namespace graphit::fir::gpu_schedule;
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
	SimpleGPUSchedule s1;
	s1.configLoadBalance(CM);
	s1.configDeduplication(ENABLED);
	s1.configFrontierCreation(UNFUSED_BITMAP);
	program->applyGPUSchedule("s1", s1);


}
}