D=/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo/vb
cp /home/kn18001/pull/vb/*.json /home/kn18001/pull/vb/*.log $D/
mkdir -p $D/results
cp -r /home/kn18001/pull/vb/results/. $D/results/
chown -R 1000:1000 $D 2>/dev/null || true
ls $D | head -20; ls $D/results | wc -l
