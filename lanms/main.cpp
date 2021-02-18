#include "lanms.h"
#include <vector>
#include <iostream>

int main()
{
    const float data[] = {0, 0, 0, 1, 1, 1, 1, 0, 1};
    std::vector<float> datavec(data, data + sizeof(data)/sizeof(float));

    std::vector<lanms::Polygon> polys = lanms::merge_quadrangle_n9(datavec.data(), datavec.size());
    std::cout << polys.size() << std::endl;

    return 0;
}
