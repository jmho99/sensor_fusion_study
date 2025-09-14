#ifndef KEYBOARD_HPP
#define KEYBOARD_HPP

#include <unistd.h>
#include <sys/select.h>
#include <sys/time.h>

namespace calib_utils
{
    bool keyboardAvailable();
}

#endif