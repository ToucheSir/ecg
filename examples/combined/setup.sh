#!/usr/bin/env bash

# Ref. https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html#Random-sources
get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

cat ../cinc*/train.json | shuf --random-source=<(get_seeded_random 2020) -o train.json
cat ../cinc*/dev.json >dev.json
cat ../cinc*/test.json >test.json

wc -l train.json dev.json test.json
