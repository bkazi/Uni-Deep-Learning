#! /bin/bash
scp ./*.py bcp4:Uni-Deep-Learning/

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  while inotifywait -e close_write ./*.py
  do
  scp ./*.py bcp4:Uni-Deep-Learning/
  done
elif [[ "$OSTYPE" == "darwin"* ]]; then
  watchman -- trigger . pyfiles '*.py' -- scp ./*.py bcp4:Uni-Deep-Learning/
else
  echo "Cannot install watchman"
fi