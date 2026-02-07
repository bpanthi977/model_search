optuna-dashboard --port 8081 "postgresql://postgres:mysecretpassword@localhost/optuna" &
PID1=$!
optuna-dashboard --port 8080 "sqlite:///logs/optuna.db" &
PID2=$!
echo "Waiting for :" $PID1 , $PID2

# Define cleanup function
cleanup() {
  echo "Caught Ctrl-C, killing processes $PID1 and $PID2..."
  kill $PID1 $PID2 2>/dev/null
  exit 1
}

# Set the trap
trap cleanup INT

# Wait for background processes to finish
wait
