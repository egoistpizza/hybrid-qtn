# Backing up queue checkpoints to your PC with Syncthing

On a Vast.ai instance without a volume, everything in `/workspace` is lost when the instance is
destroyed. With `--sync syncthing`, the queue pushes every finished run from the instance's
backup folder (`/workspace/hybrid-qtn/<tag>/`) to your PC within seconds of its backup.

**How the two machines connect:** your PC connects *out* to the instance's public address, and
the instance sends the files back over that connection. The instance never needs your PC's IP,
and you change nothing on your router. No SSH is involved. Both sides check each other's device
ID, and the traffic is encrypted.

```
Instance (send only)                                   Your PC (receive only)
/workspace/hybrid-qtn/showcase_runs/   <── PC dials ──  D:\hybrid-qtn\showcase_runs\
                                       ── files ──>
```

You do steps 1 and 3 on the PC, and steps 2, 4, 5 and 6 on the instance.

---

## 1. PC: install and prepare Syncthing

1. Download Syncthing for Windows from <https://syncthing.net/downloads/>. The **Syncthing Windows
   Setup** installer runs it in the background and starts it at login.
2. Open the Syncthing GUI at <http://127.0.0.1:8384>.
3. Copy your PC's device ID from **Actions → Show ID**. You need it in step 2.
4. Make new folders receive-only and versioned by default, so the folders you accept later get
   these settings automatically:
   - Go to **Actions → Settings → General → Edit Folder Defaults**.
   - In **Advanced → Folder Type**, choose **Receive Only**.
   - In **File Versioning**, choose **Trash Can File Versioning**. If the instance ever deletes or
     replaces a file, your PC keeps the old copy in `.stversions`.
   - Save.
5. Keep the PC awake. Go to **Settings → System → Power** and set *When plugged in, put my device
   to sleep after* to **Never**.

**What stops syncing:**

| PC state | Syncing? |
|---|---|
| Screen locked | Yes, keeps syncing |
| Asleep, hibernated or shut down | No. It catches up on wake, as long as the instance is still alive |
| Restarted but not logged in | No, unless Syncthing runs as a Windows service |

---

## 2. Instance: create the shared folder

From an SSH or tmux shell on the instance:

```bash
cd /workspace/hybrid-qtn
source venv/bin/activate
python -m scripts.backup_sync --tag showcase_runs --setup --syncthing_device <PC device ID>
```

This adds your PC to the instance's Syncthing and shares `/workspace/hybrid-qtn/showcase_runs/`
as the send-only folder `hybrid-qtn-showcase_runs`. It prints what the PC needs:

```
Sharing /workspace/hybrid-qtn/showcase_runs as Syncthing folder hybrid-qtn-showcase_runs
This machine's device ID: ABCDEFG-...  (direct address tcp://<public IP>:<port>)
syncthing hybrid-qtn-showcase_runs -> ABCDEFG offline, not seen since Syncthing started
```

Copy the **device ID** and the **direct address**. The address is
`tcp://$PUBLIC_IPADDR:$VAST_TCP_PORT_72299`, and it changes with every new instance.

---

## 3. PC: add the instance and accept the folder

1. Click **Add Remote Device** and fill it in:
   - **General:** paste the instance's device ID and give it a name, e.g. `vast-instance`.
   - **Sharing:** tick **Auto Accept**. Every queue tag is its own folder (for example the smoke
     test's `showcase_runs_e1`), and this accepts them without prompting.
   - **Advanced → Addresses:** `tcp://<public IP>:<port>, dynamic`. The direct address keeps the
     transfer fast; `dynamic` falls back to discovery and relays.
   - Save.
2. Within a minute the instance shows as **Connected**, and the folder `hybrid-qtn showcase_runs`
   appears. With Auto Accept on, it lands in your default folder path.
3. If you want the folder somewhere else (e.g. `D:\hybrid-qtn\showcase_runs`), don't tick Auto
   Accept. Instead, click **Add** on the share prompt and set the **Folder Path**. Folder Type and
   versioning come from the defaults you set in step 1.

---

## 4. Instance: check the link

```bash
python -m scripts.backup_sync --tag showcase_runs
```

| Message | Meaning |
|---|---|
| `DESKTOP-XXXX connected, up to date` | Everything is on the PC |
| `connected, 62.5% (3.0 GB to send)` | Transfer in progress |
| `offline` | The PC is asleep, Syncthing is closed, or the address is wrong |
| `has not accepted the folder yet` | Accept the share on the PC (step 3) |
| `not seen since Syncthing started` | The PC has not connected yet |

On the PC, open the remote device's details. Its address should show `tcp://...`, which is
direct. `relay://...` also works, but slower.

---

## 5. Instance: run the queue with sync on

Run a smoke test first (2 one-epoch runs, tag `showcase_runs_e1`). Then check on the PC that
`showcase_runs_e1\checkpoints\<run>\best_model.pth` arrives:

```bash
python -m scripts.run_queue --tag showcase_runs --tiers 1 --seeds 42 --epochs 1 --sync syncthing
```

Then start the full queue inside tmux, so it survives a dropped connection:

```bash
tmux new -s queue
source venv/bin/activate
python -m scripts.run_queue --tag showcase_runs --sync syncthing
```

- **`--syncthing_device`:** not needed once the PC is paired; the folder is shared with every
  paired device.
- **If something is wrong:** the queue refuses to start when Syncthing is unreachable or no device
  is paired.
- **Progress:** the `Sync:` line in `outputs/queue/showcase_runs/status.txt` shows the PC's
  progress, refreshed every 30 s.
- **Detaching from tmux:** press `Ctrl-b d`. Reattach with `tmux attach -t queue`.

---

## 6. Instance: before you destroy it

```bash
python -m scripts.backup_sync --tag showcase_runs --wait
```

Destroy the instance only after it prints `Every device has the whole folder.` and the PC shows
the folder as **Up to Date**.

---

## What arrives on the PC

```
showcase_runs\
  checkpoints\<run>\   best_model.pth, best_swa_model.pth, config.json, history.jsonl,
                       summary.json, queue_record.json
  queue\               status.txt, logs\<run>.log
```

To build the result tables on the PC, run this from a checkout of the repository:
`python -m scripts.collect_results --tag showcase_runs --backup_root D:\hybrid-qtn`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `no device to sync with` | Run step 2, or pass `--syncthing_device <PC device ID>` to the queue |
| `Syncthing API not reachable` | Run `supervisorctl status syncthing`, then `supervisorctl restart syncthing` |
| `no API key` | Use a login shell (it reads `OPEN_BUTTON_TOKEN` from `/etc/environment`), or `export SYNCTHING_API_KEY=...` |
| Instance stays offline on the PC | Check the address from step 2; Windows may ask to allow Syncthing through the firewall |
| Slow transfer | It is going through a relay; add the direct `tcp://` address (step 3) |

**Moving to a new instance:** a new instance has a new device ID and a new address. Redo steps 2
and 3.

Before launching the queue there, copy the backup folder back so the queue sees the finished runs
as done. From PowerShell on the PC:

```powershell
scp -P <ssh port> -r D:\hybrid-qtn\showcase_runs root@<public IP>:/workspace/hybrid-qtn/
```

Otherwise, runs that finished only in W&B are retrained to recover their checkpoints.
