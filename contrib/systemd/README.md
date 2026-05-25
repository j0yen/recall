# contrib/systemd

Systemd-user units for recall.

## recalld.service

Keeps the recall daemon warm in the user session.

```sh
# Install (one-time)
mkdir -p ~/.config/systemd/user
cp contrib/systemd/recalld.service ~/.config/systemd/user/

# Start now and on next login
systemctl --user daemon-reload
systemctl --user enable --now recalld.service

# Status
systemctl --user status recalld
recall daemon status
```

The unit defaults to the fastembed embedder and the user's `recall`
data root (resolved by the binary itself). Override either with a drop-in:

```sh
mkdir -p ~/.config/systemd/user/recalld.service.d
cat > ~/.config/systemd/user/recalld.service.d/override.conf <<'EOF'
[Service]
ExecStart=
ExecStart=%h/.local/bin/recalld --embedder hash --root %h/.claude/recall
EOF
systemctl --user daemon-reload
systemctl --user restart recalld
```
