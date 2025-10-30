# 🚀 How to Submit to Kaggle - Step by Step Guide

## Overview
You need to:
1. **Download the model checkpoint** from your server
2. **Upload it to Kaggle** as a dataset
3. **Create a Kaggle notebook** and run it

---

## Step 1: Download Model Checkpoint from Server

### Option A: Download via SCP (if you have SSH access)
```bash
# From your local machine:
scp ubuntu@your-server-ip:/home/ubuntu/csiro-kaggle-pasture-biomass/models/checkpoints/best_multitask_model.pth ./
```

### Option B: Download via Web Interface
If you have a web interface or can access the server files, download:
- File: `models/checkpoints/best_multitask_model.pth` (311MB)

### Option C: Use Python to download
```bash
# On your server, create a simple download script
python3 -m http.server 8000
# Then download from: http://your-server-ip:8000/models/checkpoints/best_multitask_model.pth
```

---

## Step 2: Upload Model to Kaggle Dataset

1. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets

2. **Click "New Dataset"**

3. **Fill in details:**
   - **Name**: `csiro-biomass-model`
   - **Description**: "Trained ResNet50 multi-task model for CSIRO biomass competition"
   - **License**: Choose appropriate (MIT or CC0)
   - **Visibility**: **Public** (must be public for competition use)

4. **Upload the file:**
   - Drag and drop `best_multitask_model.pth` (311MB)
   - Wait for upload to complete

5. **Click "Create"**

6. **Copy the dataset name**: It will be something like `your-username/csiro-biomass-model`

---

## Step 3: Create Kaggle Notebook

1. **Go to competition**: https://www.kaggle.com/competitions/csiro-biomass/code

2. **Click "New Notebook"**

3. **Settings** (⚙️ icon):
   - **Internet**: OFF ✅
   - **GPU**: ON ✅ (recommended)
   - **Accelerator**: T4 GPU (free) or P100 (free)

4. **Add Datasets** (📁 icon):
   - Click "+ Add input"
   - Search for: `csiro-biomass` (competition data) → Add
   - Search for: `your-username/csiro-biomass-model` (your model) → Add

---

## Step 4: Copy Code to Notebook

I'll create a standalone Python script you can copy-paste directly into Kaggle notebook.

**Copy the code from:** `notebooks/kaggle_submission_standalone.py` (I'll create this)

Or follow the instructions in the notebook file.

---

## Step 5: Run and Submit

1. **Click "Run All"** (or press Shift+Enter)

2. **Wait for completion** (should take 1-2 minutes)

3. **Check output**: Should see "✅ Submission file created: submission.csv"

4. **Save Version**:
   - Click "Save Version" button
   - Select "Save & Run All"
   - Add description: "Multi-task ResNet50 model submission"

5. **Submit**:
   - After save completes, click **"Submit"** button
   - The button will be active once the notebook runs successfully

---

## Quick Checklist

- [ ] Downloaded `best_multitask_model.pth` from server
- [ ] Created Kaggle dataset with model checkpoint
- [ ] Created new Kaggle notebook
- [ ] Added competition dataset (`csiro-biomass`)
- [ ] Added model dataset (`your-username/csiro-biomass-model`)
- [ ] Set Internet: OFF, GPU: ON
- [ ] Copied code into notebook
- [ ] Ran notebook successfully
- [ ] Saved version
- [ ] Clicked Submit button

---

## Need Help?

If you're stuck, tell me:
- Do you have SSH access to the server?
- Can you access the server files via a web interface?
- Do you have the model file downloaded already?

I'll help you with the specific step you're on!

