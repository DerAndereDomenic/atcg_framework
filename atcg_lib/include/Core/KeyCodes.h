#pragma once

/**
 *  @brief Keyboard key IDs.
 *  @note These are the ATCG keycodes, copied for compatibility.
 *
 *
 *  These key codes are inspired by the _USB HID Usage Tables v1.12_ (p. 53-60),
 *  but re-arranged to map to 7-bit ASCII for printable keys (function keys are
 *  put in the 256+ range).
 *
 *  The naming of the key codes follow these rules:
 *   - The US keyboard layout is used
 *   - Names of printable alphanumeric characters are used (e.g. "A", "R",
 *     "3", etc.)
 *   - For non-alphanumeric characters, Unicode:ish names are used (e.g.
 *     "COMMA", "LEFT_SQUARE_BRACKET", etc.). Note that some names do not
 *     correspond to the Unicode standard (usually for brevity)
 *   - Keys that lack a clear US mapping are named "WORLD_x"
 *   - For non-printable keys, custom names are used (e.g. "F4",
 *     "BACKSPACE", etc.)
 */

/* The unknown key */
#define ATCG_KEY_UNKNOWN -1

/* Printable keys */
#define ATCG_KEY_SPACE         32
#define ATCG_KEY_APOSTROPHE    39 /* ' */
#define ATCG_KEY_COMMA         44 /* , */
#define ATCG_KEY_MINUS         45 /* - */
#define ATCG_KEY_PERIOD        46 /* . */
#define ATCG_KEY_SLASH         47 /* / */
#define ATCG_KEY_0             48
#define ATCG_KEY_1             49
#define ATCG_KEY_2             50
#define ATCG_KEY_3             51
#define ATCG_KEY_4             52
#define ATCG_KEY_5             53
#define ATCG_KEY_6             54
#define ATCG_KEY_7             55
#define ATCG_KEY_8             56
#define ATCG_KEY_9             57
#define ATCG_KEY_SEMICOLON     59 /* ; */
#define ATCG_KEY_EQUAL         61 /* = */
#define ATCG_KEY_A             65
#define ATCG_KEY_B             66
#define ATCG_KEY_C             67
#define ATCG_KEY_D             68
#define ATCG_KEY_E             69
#define ATCG_KEY_F             70
#define ATCG_KEY_G             71
#define ATCG_KEY_H             72
#define ATCG_KEY_I             73
#define ATCG_KEY_J             74
#define ATCG_KEY_K             75
#define ATCG_KEY_L             76
#define ATCG_KEY_M             77
#define ATCG_KEY_N             78
#define ATCG_KEY_O             79
#define ATCG_KEY_P             80
#define ATCG_KEY_Q             81
#define ATCG_KEY_R             82
#define ATCG_KEY_S             83
#define ATCG_KEY_T             84
#define ATCG_KEY_U             85
#define ATCG_KEY_V             86
#define ATCG_KEY_W             87
#define ATCG_KEY_X             88
#define ATCG_KEY_Y             89
#define ATCG_KEY_Z             90
#define ATCG_KEY_LEFT_BRACKET  91  /* [ */
#define ATCG_KEY_BACKSLASH     92  /* \ */
#define ATCG_KEY_RIGHT_BRACKET 93  /* ] */
#define ATCG_KEY_GRAVE_ACCENT  96  /* ` */
#define ATCG_KEY_WORLD_1       161 /* non-US #1 */
#define ATCG_KEY_WORLD_2       162 /* non-US #2 */

/* Function keys */
#define ATCG_KEY_ESCAPE        256
#define ATCG_KEY_ENTER         257
#define ATCG_KEY_TAB           258
#define ATCG_KEY_BACKSPACE     259
#define ATCG_KEY_INSERT        260
#define ATCG_KEY_DELETE        261
#define ATCG_KEY_RIGHT         262
#define ATCG_KEY_LEFT          263
#define ATCG_KEY_DOWN          264
#define ATCG_KEY_UP            265
#define ATCG_KEY_PAGE_UP       266
#define ATCG_KEY_PAGE_DOWN     267
#define ATCG_KEY_HOME          268
#define ATCG_KEY_END           269
#define ATCG_KEY_CAPS_LOCK     280
#define ATCG_KEY_SCROLL_LOCK   281
#define ATCG_KEY_NUM_LOCK      282
#define ATCG_KEY_PRINT_SCREEN  283
#define ATCG_KEY_PAUSE         284
#define ATCG_KEY_F1            290
#define ATCG_KEY_F2            291
#define ATCG_KEY_F3            292
#define ATCG_KEY_F4            293
#define ATCG_KEY_F5            294
#define ATCG_KEY_F6            295
#define ATCG_KEY_F7            296
#define ATCG_KEY_F8            297
#define ATCG_KEY_F9            298
#define ATCG_KEY_F10           299
#define ATCG_KEY_F11           300
#define ATCG_KEY_F12           301
#define ATCG_KEY_F13           302
#define ATCG_KEY_F14           303
#define ATCG_KEY_F15           304
#define ATCG_KEY_F16           305
#define ATCG_KEY_F17           306
#define ATCG_KEY_F18           307
#define ATCG_KEY_F19           308
#define ATCG_KEY_F20           309
#define ATCG_KEY_F21           310
#define ATCG_KEY_F22           311
#define ATCG_KEY_F23           312
#define ATCG_KEY_F24           313
#define ATCG_KEY_F25           314
#define ATCG_KEY_KP_0          320
#define ATCG_KEY_KP_1          321
#define ATCG_KEY_KP_2          322
#define ATCG_KEY_KP_3          323
#define ATCG_KEY_KP_4          324
#define ATCG_KEY_KP_5          325
#define ATCG_KEY_KP_6          326
#define ATCG_KEY_KP_7          327
#define ATCG_KEY_KP_8          328
#define ATCG_KEY_KP_9          329
#define ATCG_KEY_KP_DECIMAL    330
#define ATCG_KEY_KP_DIVIDE     331
#define ATCG_KEY_KP_MULTIPLY   332
#define ATCG_KEY_KP_SUBTRACT   333
#define ATCG_KEY_KP_ADD        334
#define ATCG_KEY_KP_ENTER      335
#define ATCG_KEY_KP_EQUAL      336
#define ATCG_KEY_LEFT_SHIFT    340
#define ATCG_KEY_LEFT_CONTROL  341
#define ATCG_KEY_LEFT_ALT      342
#define ATCG_KEY_LEFT_SUPER    343
#define ATCG_KEY_RIGHT_SHIFT   344
#define ATCG_KEY_RIGHT_CONTROL 345
#define ATCG_KEY_RIGHT_ALT     346
#define ATCG_KEY_RIGHT_SUPER   347
#define ATCG_KEY_MENU          348

#define ATCG_KEY_LAST ATCG_KEY_MENU

#define ATCG_MOUSE_BUTTON_1      0
#define ATCG_MOUSE_BUTTON_2      1
#define ATCG_MOUSE_BUTTON_3      2
#define ATCG_MOUSE_BUTTON_4      3
#define ATCG_MOUSE_BUTTON_5      4
#define ATCG_MOUSE_BUTTON_6      5
#define ATCG_MOUSE_BUTTON_7      6
#define ATCG_MOUSE_BUTTON_8      7
#define ATCG_MOUSE_BUTTON_LAST   ATCG_MOUSE_BUTTON_8
#define ATCG_MOUSE_BUTTON_LEFT   ATCG_MOUSE_BUTTON_1
#define ATCG_MOUSE_BUTTON_RIGHT  ATCG_MOUSE_BUTTON_2
#define ATCG_MOUSE_BUTTON_MIDDLE ATCG_MOUSE_BUTTON_3