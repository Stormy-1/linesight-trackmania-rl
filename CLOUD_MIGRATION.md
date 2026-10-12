# ☁️ Cloud Migration Guide — Project Linesight

> **Trigger:** At 10M frames, stop local training and move to a cloud Windows GPU VM.  
> **Why:** Local laptop is bottlenecked (2 collectors, thermal throttling, shared resources). A cloud T4/L4 GPU VM with more RAM and CPU cores can run 4-6+ collectors simultaneously, dramatically accelerating frames/hour.

---

## 🚨 Why NOT Google Colab?

The `PROJECT_STORY.md` originally suggested Colab, but **Colab fundamentally cannot run this project**:

| Requirement | Colab Provides | Verdict |
|---|---|---|
| TrackMania Forever (`.exe`) | Linux-only, no Windows apps | ❌ |
| TMInterface (hooks into TmForever) | No Wine, no display server | ❌ |
| Visible window for frame capture | Headless — no display | ❌ |
| GPU for neural network training | ✅ T4 GPU available | ✅ |

The training loop is **not a pure GPU job** — each collector process spawns a live `TmForever.exe` instance, connects via socket (`tminterface2.py`), captures frames via `grab_screen()`, and sends inputs. This requires a full Windows desktop (or Linux + Wine + Xvfb, which is fragile and unsupported on Colab).

**Solution: A Windows GPU VM you can RDP into.**

---

## 🎓 Free Student Credits (Your Best Options)

You have a verified student ID + GitHub Student Developer Pack. Here's how to get cloud compute for free:

### 🏆 Option 1: Azure for Students — $100 FREE (Primary Recommendation)

**How to claim:**
1. Go to [azure.microsoft.com/en-us/free/students/](https://azure.microsoft.com/en-us/free/students/)
2. Sign in with your **student email** (or link via GitHub Student Pack)
3. Verify student status → **$100 credit activated, NO credit card required!**

**Will it be enough?**

| VM Type | Cost/hr | Hours for $100 | Days (24/7) |
|---|---|---|---|
| `NC4as_T4_v3` (GPU, 4 vCPU, 28 GB) | ~$0.53 | **189 hrs** | ~7.8 days |
| `NC4as_T4_v3` **Spot** | ~$0.15 | **666 hrs** | ~27 days |
| `NC8as_T4_v3` (GPU, 8 vCPU, 56 GB) | ~$0.95 | **105 hrs** | ~4.3 days |
| `D8s_v3` (CPU-only fallback, 8 vCPU) | ~$0.38 | **263 hrs** | ~11 days |

**For 24 hours of training, you'd spend ~$4-13. $100 is WAY more than enough.**

> ⚠️ **The GPU Quota Catch:** Azure Student subscriptions default to **0 GPU cores**. After claiming credits:
> 1. Go to **Azure Portal → Subscriptions → Usage + Quotas**
> 2. Check if `NCasT4_v3` family has any quota
> 3. If it's 0, submit a **quota increase request** — explain it's for a university ML/RL research project
> 4. Some students get approved, some don't. If denied, use the CPU-only fallback (`D8s_v3`) or try Option 2

> 💡 **If GPU quota is denied:** A CPU-only `D8s_v3` (8 vCPU, 32 GB RAM, ~$0.38/hr) can still run 4 collectors. The neural network trains on CPU instead of GPU — slower per batch, but the real bottleneck is TrackMania rollout speed anyway. It'll still be a significant upgrade over your laptop.

### 🥈 Option 2: GCP $300 Free Trial (Backup — requires credit card)

**The trick:** GCP's free trial blocks GPUs and Windows VMs. But if you **upgrade to a paid account**, your $300 credits carry over AND all restrictions lift.

**How to claim:**
1. Go to [cloud.google.com/free](https://cloud.google.com/free)
2. Sign up → get **$300 credit for 90 days**
3. **Immediately upgrade** to a paid account (your $300 stays!)
4. Request GPU quota increase (usually approved after upgrade)
5. Set a **billing alert at $290** so you stop before real charges hit

| VM Type | Cost/hr | Hours for $300 | Days (24/7) |
|---|---|---|---|
| `n1-standard-8` + T4 GPU | ~$0.85 | **353 hrs** | ~14.7 days |
| `n1-standard-8` + T4 **Spot** | ~$0.28 | **1071 hrs** | ~44 days |

> ⚠️ **Risk:** After upgrading, if you forget to shut down the VM, you **will** get charged real money once credits run out. Set billing alerts! But with discipline, this is the most generous option — **$300 is absurd overkill for your needs.**

### ❌ Options That Won't Work

| Option | Why Not |
|---|---|
| AWS Educate | Learning labs only, no GPU instances |
| AWS Free Tier | No GPUs in free tier |
| GCP Free Trial (without upgrade) | Blocks GPUs AND Windows VMs |
| Vast.ai / RunPod | Linux-only, no Windows |

### 🎯 Recommended Strategy

1. **Try Azure for Students first** — $100, no credit card, zero risk
2. If Azure GPU quota is denied → **GCP $300 with upgrade** — more credits, needs credit card but set alerts
3. Either way, **$100 covers 24+ hours of GPU compute easily**

---

## 💰 Provider Comparison (Full Reference)

| Provider | Instance | vCPUs | RAM | GPU | ~Cost/hr (Windows) | Notes |
|---|---|---|---|---|---|---|
| **Azure** ⭐ | `NC4as_T4_v3` | 4 | 28 GB | 1x T4 | **~$0.53** | **Student credits!** Best RAM. Spot ~$0.15/hr |
| **Azure** ⭐ | `NC8as_T4_v3` | 8 | 56 GB | 1x T4 | **~$0.95** | 4 collectors. Spot ~$0.30/hr |
| **GCP** | `n1-standard-4` + T4 | 4 | 15 GB | 1x T4 | **~$0.75** | $300 free trial (upgrade required for GPU) |
| **GCP** | `n1-standard-8` + T4 | 8 | 30 GB | 1x T4 | **~$0.85** | 4 collectors |
| **AWS EC2** | `g4dn.xlarge` | 4 | 16 GB | 1x T4 | **~$0.71** | No student credits. Spot ~$0.20/hr |
| **AWS EC2** | `g4dn.2xlarge` | 8 | 32 GB | 1x T4 | **~$1.12** | 4 collectors. Spot ~$0.40/hr |

> **💡 Budget tip:** Use **Spot/Preemptible instances** for 60-90% savings. Training auto-saves weights (`weights1.torch`) continuously, so preemption only loses the current rollout — you resume instantly.

> **⚠️ Scaling collectors:** Each TrackMania collector uses ~2 vCPUs and ~3-4 GB RAM. A 4-vCPU instance fits 2 collectors (same as your laptop). **For a real speedup, go 8 vCPU minimum → 4 collectors.**

---

## 📋 Pre-Migration Checklist (Do Before Launching the VM)

### 1. Wait for 10M Frames & Stop Training
```powershell
# Check current frame count
& "$HOME\miniconda3\envs\linesight\python.exe" -c "import joblib; s = joblib.load('save/hocko_v1/accumulated_stats.joblib'); print(f'Frames: {s[\"cumul_number_frames_played\"]:,}')"
```
- When frames ≥ 10M, gracefully stop training with `Ctrl+C`
- Training will finish the current rollout and save

### 2. Create a Final Backup
```powershell
# Backup the 10M-frame checkpoint
Copy-Item -Path "save\hocko_v1" -Destination "save\hocko_v1_10M_local_backup" -Recurse
```

### 3. What to Transfer

| Component | Size | Transfer? |
|---|---|---|
| Code + configs + reference lines | ~13 MB | ✅ Yes |
| `save/hocko_v1/` (active weights) | ~204 MB | ✅ Yes |
| `save/hocko_v1_pre_tuning_backup_54_24s/` | ~203 MB | ✅ Yes (safety net) |
| `save/hocko_v1_foundational_checkpoint/` | ~202 MB | ✅ Yes (95% generalization base) |
| `tensorboard/` logs | ~45 MB | ⚠️ Optional (nice for continuity) |
| `save/hocko_v1_10M_local_backup/` | ~204 MB | ❌ Keep local only |
| **Total to upload** | **~665 MB** | Fits in one zip |

### 4. Create the Transfer Archive
```powershell
# From the TRACK directory — exclude local-only backup and caches
cd C:\Code_Stuff\TRACK

# Option A: Simple zip (PowerShell built-in)
Compress-Archive -Path * -DestinationPath "$HOME\Desktop\linesight_cloud_transfer.zip" -Force

# Option B: If you have 7-Zip (better compression)
& "C:\Program Files\7-Zip\7z.exe" a "$HOME\Desktop\linesight_cloud_transfer.7z" * -xr!__pycache__ -xr!.git -xr!hocko_v1_10M_local_backup
```

> **📝 Note:** If the zip is too large for easy upload, you can skip `tensorboard/` and the backup checkpoints. **The only essential files are**: the codebase + `save/hocko_v1/` (~217 MB total).

---

## 🖥️ VM Setup (Step-by-Step for Azure)

> If using GCP instead, the core steps are identical — just swap instance types and use GCP Console instead of Azure Portal.

### Step 1: Launch the Instance

1. Go to **[Azure Portal](https://portal.azure.com) → Virtual Machines → Create**
2. **Subscription:** `Azure for Students`
3. **Resource Group:** Create new → `linesight-rg`
4. **VM Name:** `linesight-training`
5. **Region:** Pick one close to you (e.g., `Central India` or `East US`) — check that `NCasT4_v3` is available in that region
6. **Image:** `Windows Server 2022 Datacenter`
7. **Size:** `NC4as_T4_v3` (4 vCPU, 28 GB, 1x T4) — or `NC8as_T4_v3` for 4 collectors
   - If GPU sizes aren't listed, you need the quota increase (see Student Credits section above)
   - **Fallback:** `Standard_D8s_v3` (8 vCPU, 32 GB, CPU-only)
8. **Username/Password:** Set an admin account (you'll RDP with this)
9. **Inbound Ports:** Allow **RDP (3389)**
10. **Disk:** 100 GB Premium SSD
11. **Spot Instance:** Under "Advanced" tab, check "Run with Azure Spot discount" for ~70% savings
12. **Review + Create!**

### Step 2: Connect via RDP

1. Wait ~5 min for Windows to boot
2. In Azure Portal → Your VM → **Connect → RDP**
3. Download the `.rdp` file
4. Open it → enter the username/password you set
5. Connect via Remote Desktop

### Step 3: Install Dependencies on the VM

Open PowerShell as Administrator on the VM:

```powershell
# 1. Install Miniconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "$HOME\miniconda.exe"
Start-Process -FilePath "$HOME\miniconda.exe" -ArgumentList "/S /D=$HOME\miniconda3" -Wait

# 2. Create the conda environment (CRITICAL: Python 3.11, NOT 3.13!)
& "$HOME\miniconda3\Scripts\conda.exe" create -n linesight python=3.11.16 -y

# 3. Install PyTorch with CUDA support
& "$HOME\miniconda3\envs\linesight\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 💡 **If using CPU-only fallback (D8s_v3):** Install CPU-only PyTorch instead:
> ```powershell
> & "$HOME\miniconda3\envs\linesight\python.exe" -m pip install torch torchvision
> ```

### Step 4: Install TrackMania Forever + TMInterface

> ⛔ **This is the most finicky step. Follow exactly.**

1. **Download TrackMania Nations Forever** (free) — install it manually via the `.exe` installer
2. **Download TMInterface v2.1.0** from the [TMInterface releases page](https://donadigo.com/tminterface)
3. **Download TMLoader** — install to `%LOCALAPPDATA%\TMLoader\TMLoader.exe`
4. **Configure TMLoader:**
   - Launch TMLoader
   - Create a profile named `default`
   - Point it to the TMInterface executable
   - **Close TMLoader completely** after configuration
5. **Set up the TMNF account:**
   - Launch TrackMania once manually
   - Create/login with username: `stormy`
   - Close the game

### Step 5: Upload & Extract the Project

```powershell
# Option A: Direct upload via RDP (drag-and-drop — works with Windows Remote Desktop)
# Just drag linesight_cloud_transfer.zip from your laptop into the RDP window

# Option B: Use Azure Blob Storage as intermediary
# (Install Azure CLI on VM first: winget install Microsoft.AzureCLI)
az storage blob download --account-name <your-storage> --container <container> --name linesight_cloud_transfer.zip --file C:\Code_Stuff\TRACK.zip
Expand-Archive -Path C:\Code_Stuff\TRACK.zip -DestinationPath C:\Code_Stuff\TRACK

# Option C: Upload to Google Drive, download via browser on the VM
```

### Step 6: Install Python Dependencies

```powershell
cd C:\Code_Stuff\TRACK
& "$HOME\miniconda3\envs\linesight\python.exe" -m pip install -r requirements_pip.txt
```

> ⚠️ **Critical:** Ensure `torchrl==0.2.1` and `tensordict==0.2.1` install correctly. Newer versions break with `MemmapTensor` import errors. If pip tries to install newer versions, force:
> ```powershell
> & "$HOME\miniconda3\envs\linesight\python.exe" -m pip install torchrl==0.2.1 tensordict==0.2.1
> ```

### Step 7: Update Configuration for Cloud

Edit `config_files/user_config.py` on the VM:

```python
# Update paths for the cloud VM (no OneDrive on the VM!)
target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
trackmania_base_path = Path(os.path.expanduser("~")) / "Documents" / "TrackMania"

# Update TMLoader path (verify this matches your install)
windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
```

Edit `config_files/config.py` — **scale up collectors:**

```python
# CLOUD CONFIG: Scale from 2 → 4 collectors (8 vCPU VM can handle this)
gpu_collectors_count = 4  # Was 2 on laptop
```

### Step 8: Verify GPU & Launch Training

```powershell
# Verify GPU is visible
nvidia-smi  # Should show the T4 GPU

# IMPORTANT: Make sure TMLoader GUI is CLOSED before running!
cd C:\Code_Stuff\TRACK
& "$HOME\miniconda3\envs\linesight\python.exe" scripts/train.py
```

You should see:
- 4 TrackMania windows spawning
- "Found Trackmania process id" messages  
- "Initialize connection to TMInterface" → successful connections
- Training batches starting

> **💡 Tip:** You can disconnect from RDP and the training continues running (the desktop session stays alive). Reconnect anytime to check progress. Just don't **log out** — that kills the desktop session and the game windows.

---

## 🔧 Cloud-Specific Config Changes

### Scaling Collectors (The Main Speed Win)

The number of collectors you can run depends on your VM's CPU and RAM:

| VM Size | vCPUs | RAM | Max Collectors | Expected Speedup vs Laptop |
|---|---|---|---|---|
| `g4dn.xlarge` (4 vCPU) | 4 | 16 GB | 2 | ~1x (same as laptop) |
| `g4dn.2xlarge` (8 vCPU) | 8 | 32 GB | **4** | **~2x** |
| `g4dn.4xlarge` (16 vCPU) | 16 | 64 GB | **6-8** | ~3-4x |

In `config.py` / `config_copy.py`:
```python
gpu_collectors_count = 4  # Adjust based on your VM
```

Each collector needs its own TMI port (auto-assigned as `base_tmi_port + 0`, `+1`, `+2`, `+3`).

### Frame Count Continues from Local

The `accumulated_stats.joblib` file travels with the save — frame count, alltime best, rolling mean all carry over seamlessly. The LR schedule, epsilon schedule, memory size schedule, etc. will continue exactly where they left off at ~10M frames.

---

## ⚠️ Gotchas & Lookouts

### Critical Issues

1. **RDP Disconnect ≠ Logout**
   - **Disconnect** (close RDP window): Desktop session stays alive ✅
   - **Sign out / Log off**: Kills the session, kills all game windows ❌
   - **Always disconnect, never sign out**

2. **TMLoader Must Be Closed**
   - Same rule as local: TMLoader GUI must NOT be running when you start `train.py`
   - If the ModLoader GUI is open, the subprocess hands execution off to the background task and exits, causing endless `Connection to TMInterface unsuccessful` messages

3. **OneDrive Paths Don't Exist on VM**
   - Your local `user_config.py` uses `OneDrive/Documents/...`
   - The VM will use plain `Documents/...`
   - **Must update `user_config.py`** before first run (see Step 7)

4. **Spot Instance Preemption**
   - Azure can reclaim Spot VMs when it needs capacity
   - Training auto-saves weights continuously, so you lose at most one rollout
   - **Recovery:** Just create a new Spot VM and resume from `save/hocko_v1/`

5. **GPU Driver Verification**
   ```powershell
   nvidia-smi  # Should show the T4 GPU
   ```
   Azure NCasT4_v3 images usually come with NVIDIA drivers pre-installed. If `nvidia-smi` fails, install from https://www.nvidia.com/drivers

### Performance Gotchas

6. **RDP Display Resolution**
   - TrackMania captures frames at the game window resolution
   - Ensure each TM window is not minimized (the code handles this, but verify)
   - RDP's default resolution should be fine (1920x1080)

7. **Windows Firewall**
   - TMInterface communicates via localhost sockets (ports 8478-8481+)
   - These are loopback connections, so firewall shouldn't block them
   - If you see `Connection to TMInterface unsuccessful`, check firewall rules

8. **Storage Space**
   - `best_runs/` folder grows over time (saves replay data for each PB)
   - 100 GB SSD gives plenty of headroom
   - Monitor with: `Get-ChildItem save -Recurse | Measure-Object -Property Length -Sum`

---

## 📊 Monitoring Training from Your Laptop

### Option A: Quick Stats Check via RDP
Just reconnect to the VM via RDP and run:
```powershell
cd C:\Code_Stuff\TRACK
& "$HOME\miniconda3\envs\linesight\python.exe" -c "import joblib; s = joblib.load('save/hocko_v1/accumulated_stats.joblib'); print(f'Frames: {s[\"cumul_number_frames_played\"]:,}'); print(f'Best: {s[\"alltime_min_ms\"][\"hock\"]/1000:.3f}s'); print(f'Mean: {s[\"rolling_mean_ms\"][\"hock\"]/1000:.2f}s')"
```

### Option B: TensorBoard via RDP
```powershell
# On the VM, launch TensorBoard
& "$HOME\miniconda3\envs\linesight\python.exe" -m tensorboard --logdir tensorboard --port 6006 --bind_all
```
Then open `http://<VM-PUBLIC-IP>:6006` in your laptop browser (make sure port 6006 is allowed in the Azure NSG / firewall rules).

---

## 💸 Cost Management

### Estimated Training Costs (Azure for Students — $100 budget)

| Duration | `NC4as_T4_v3` Spot (~$0.15/hr) | `NC4as_T4_v3` On-Demand (~$0.53/hr) | `NC8as_T4_v3` On-Demand (~$0.95/hr) |
|---|---|---|---|
| 8 hours | ~$1.20 | ~$4.24 | ~$7.60 |
| 24 hours | ~$3.60 | ~$12.72 | ~$22.80 |
| 72 hours (3 days) | ~$10.80 | ~$38.16 | ~$68.40 |
| **Max hours on $100** | **~666 hrs** | **~189 hrs** | **~105 hrs** |

### Cost-Saving Tips

1. **Use Spot instances** — saves 60-70%
2. **STOP the VM when not training** — you're billed per-minute when running. Go to Azure Portal → VM → **Stop** (not just disconnect RDP!)
3. **Set a budget alert** in Azure Portal → Cost Management → Budgets → Create at $20, $50, $80
4. **Deallocate, don't just Stop** — "Stop" in Azure Portal deallocates resources and stops billing. Shutting down from inside Windows does NOT stop billing!
5. **Delete the VM when done** — disk storage costs ~$0.10/GB/month even when VM is stopped

---

## 🔄 Bringing Results Back to Local

When you're done with a cloud training run:

```powershell
# On the VM — zip the results
Compress-Archive -Path "C:\Code_Stuff\TRACK\save\hocko_v1" -DestinationPath "$HOME\Desktop\hocko_v1_cloud_results.zip"

# Upload to Google Drive via browser on the VM, then download on your laptop
# OR drag-and-drop from RDP session back to your local machine
```

---

## 🎯 Summary: Migration Day Playbook

```
 LOCAL LAPTOP                          AZURE CLOUD VM
 ────────────                          ──────────────
 0. Claim Azure for Students $100
 1. Hit 10M frames → Ctrl+C            
 2. Backup save/hocko_v1               
 3. Zip TRACK folder (~665 MB)         
 4. Upload to Google Drive        →    5. Download on VM via browser
                                       6. Install Miniconda + Python 3.11.16
                                       7. Install pip deps (torchrl==0.2.1!)
                                       8. Install TrackMania + TMInterface + TMLoader
                                       9. Update user_config.py (remove OneDrive paths)
                                       10. Set gpu_collectors_count = 4
                                       11. Close TMLoader GUI
                                       12. Run: python scripts/train.py
                                       13. Verify TM windows + training batches
                                       14. Disconnect RDP (don't log out!)
 15. Monitor via RDP reconnect         ← Training runs autonomously
 16. When done: download results       ← Upload to Google Drive
 17. STOP VM in Azure Portal!          ← Don't forget or credits drain!
```

> **⚡ The single biggest speed gain** is going from 2 collectors → 4+ collectors. Everything else (GPU speed, CPU speed) is secondary. The T4 GPU is comparable to your laptop GPU for the neural network side — the bottleneck has always been data collection speed (how fast TrackMania can run rollouts).

