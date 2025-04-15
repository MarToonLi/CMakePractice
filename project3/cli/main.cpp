#include "main.h"


int main()
{
	SetConsoleOutputCP(CP_UTF8);  // allow the chinese log to show.
	SetConsoleCP(CP_UTF8);
	system("chcp 65001");

	LOGD("SSS 测试");

	int a = yolo_min();

	return 1;
}
