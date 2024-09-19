#ifndef OST_UTILS_HPP
#define OST_UTILS_HPP
#include <vector>

struct OSTTemplateData{
  int crop_w;
  int crop_h;
  int template_input_size;
  std::vector<unsigned char> templateData;
  bool initialized=false;
};
#endif
