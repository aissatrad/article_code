def android_components(app, c_name):
    res = app.get_manifest().get("application").get(c_name)
    res = res if type(res) is list else [res]
    return [r.get("@android:name") for r in res]


def perms_arr(perms):
    file_perms = open("perms.txt", "r")
    all_perms = [p.strip("\n") for p in file_perms.readlines()]
    file_perms.close()
    fperms = []
    for p in all_perms:
        if p in perms:
            fperms.append(1)
        else:
            fperms.append(0)
    return fperms

