#include "calib_utils/keyboard.hpp"

#include <unistd.h>
#include <sys/select.h>
#include <sys/time.h>

namespace calib_utils
{
    bool keyboardAvailable()
    {
        struct timeval tv{0L, 0L};
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        return select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0;
    }
}