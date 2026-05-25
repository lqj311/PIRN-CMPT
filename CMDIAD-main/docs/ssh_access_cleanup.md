# SSH access cleanup

Date: 2026-05-25

This project used a temporary dedicated SSH key for Codex access to the remote server.

## Local key files to delete

Run on Windows PowerShell:

```powershell
Remove-Item C:\Users\User\.ssh\id_ed25519_pirn_seetacloud -Force
Remove-Item C:\Users\User\.ssh\id_ed25519_pirn_seetacloud.pub -Force
```

## Remote authorization to remove

The public key installed on the server has this comment:

```text
codex-pirn-seetacloud
```

Run on the server:

```bash
sed -i '/codex-pirn-seetacloud/d' ~/.ssh/authorized_keys
```

## Remote server

```text
ssh -p 23609 root@connect.cqa1.seetacloud.com
project: /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
```

After cleanup, key-based access from this machine should no longer work.
