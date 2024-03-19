#ifndef DETECTION_HPP
#define DETECTION_HPP

namespace arm {
namespace app {
namespace pg_object_detection {

    struct Box {
        float x;
        float y;
        float w;
        float h;
    };

    struct Position {
        float x;
        float y;
    };

    struct Detection {
        Box bbox;
        Position position;
        float confidence;
        int cls;
    };

} /* namespace pg_object_detection */
} /* namespace app */
} /* namespace arm */

#endif // DETECTION_HPP