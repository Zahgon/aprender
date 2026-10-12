# fixture_quiet_wiring.awk -- fixture_quiet.sh --check-wiring's reader (#5063).
# Input: GitHub workflow files. A step is a `- ` item plus the more-indented
# lines under it; comment lines and trailing ` #` comments are dropped, and a
# backslash is read as a space so a continued line still names its verb.
# Prints one FAIL line per fat_driver.py self-test step not wrapped in
# fixture_quiet.sh and per run or wait step that is; NONE when no self-test
# step was seen at all.
function flush(   v, verbs, n, i, w) {
    if (step == "") return
    w = (step ~ /fixture_quiet[.]sh/)
    n = split("self-test run wait", verbs, " ")
    for (i = 1; i <= n; i++) {
        v = verbs[i]
        if (step !~ ("fat_driver[.]py[ \t]+" v "([^a-z-]|$)")) continue
        if (v == "self-test") {
            selftests++
            if (!w) printf "FAIL %s:%d: a fat_driver.py self-test step is not wrapped in fixture_quiet.sh\n", sfile, sline
        } else if (w) {
            printf "FAIL %s:%d: a fat_driver.py %s step is wrapped in fixture_quiet.sh: its real failures would read as fixture text\n", sfile, sline, v
        }
    }
    step = ""
}
FNR == 1 { flush() }
/^[ \t]*$/ || /^[ \t]*#/ { next }
{
    line = $0; sub(/[ \t]#.*$/, "", line); gsub(/\\/, " ", line)
    match(line, /^ */); ind = RLENGTH
    if (step != "" && ind > sind) { step = step " " line; next }
    flush()
    if (line ~ /^ *- /) { step = line; sind = ind; sline = FNR; sfile = FILENAME }
}
END { flush(); if (selftests == 0) print "NONE" }
