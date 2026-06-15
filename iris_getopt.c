/*
 * Minimal getopt_long implementation for Windows (MSVC).
 * Not needed on POSIX/macOS where the native getopt_long is used.
 */

#ifdef _WIN32

#include "iris_platform.h"
#include <stdio.h>
#include <string.h>

int iris_optind = 1;
int iris_optopt = 0;
char *iris_optarg = NULL;

int iris_getopt_long(int argc, char * const argv[],
                     const char *shortopts,
                     const iris_option_t *longopts,
                     int *longindex) {
    (void)longindex;

    if (iris_optind >= argc) return -1;

    const char *arg = argv[iris_optind];

    /* End of options */
    if (arg[0] != '-') return -1;
    if (arg[1] == '\0') return -1;

    /* -- signals end of options */
    if (arg[1] == '-' && arg[2] == '\0') {
        iris_optind++;
        return -1;
    }

    /* Long option */
    if (arg[1] == '-') {
        const char *name = arg + 2;
        const char *eq = strchr(name, '=');
        size_t namelen = eq ? (size_t)(eq - name) : strlen(name);

        for (int i = 0; longopts[i].name; i++) {
            if (strlen(longopts[i].name) == namelen &&
                strncmp(longopts[i].name, name, namelen) == 0) {
                iris_optopt = longopts[i].val;
                if (longopts[i].has_arg == no_argument) {
                    iris_optind++;
                    if (longopts[i].flag) {
                        *longopts[i].flag = longopts[i].val;
                        return 0;
                    }
                    return longopts[i].val;
                }
                if (eq) {
                    iris_optarg = (char*)(eq + 1);
                    iris_optind++;
                } else if (iris_optind + 1 < argc) {
                    iris_optind++;
                    iris_optarg = argv[iris_optind];
                    iris_optind++;
                } else {
                    fprintf(stderr, "Option --%s requires an argument\n", name);
                    return '?';
                }
                if (longopts[i].flag) {
                    *longopts[i].flag = longopts[i].val;
                    return 0;
                }
                return longopts[i].val;
            }
        }
        fprintf(stderr, "Unknown option: %s\n", arg);
        iris_optind++;
        return '?';
    }

    /* Short option */
    const char *opt = arg + 1;
    iris_optopt = (unsigned char)*opt;

    const char *sp = strchr(shortopts, *opt);
    if (!sp) {
        fprintf(stderr, "Unknown option: -%c\n", *opt);
        iris_optind++;
        return '?';
    }

    if (sp[1] == ':') {
        if (opt[1] != '\0') {
            iris_optarg = (char*)(opt + 1);
            iris_optind++;
        } else if (iris_optind + 1 < argc) {
            iris_optind++;
            iris_optarg = argv[iris_optind];
            iris_optind++;
        } else {
            fprintf(stderr, "Option -%c requires an argument\n", *opt);
            return ':';
        }
    } else {
        iris_optarg = NULL;
        iris_optind++;
    }

    return iris_optopt;
}

#endif /* _WIN32 */
