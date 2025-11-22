import { useEffect, useState } from "react";
import { addPrankTrigger, createPrank, fetchPrank } from "./api";
import "./index.css";

interface TriggerRow {
  id: string;
  triggerText: string;
  file: File | null;
  previewUrl?: string;
  existing?: boolean;
}

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

interface PrankCreateAppProps {
  initialSlug?: string;
  requirePassword?: boolean;
  adminPassword?: string;
}

export default function PrankCreateApp({ initialSlug, requirePassword, adminPassword }: PrankCreateAppProps) {
  const [title, setTitle] = useState("");
  const [rows, setRows] = useState<TriggerRow[]>([{ id: makeId(), triggerText: "", file: null }]);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [prankId, setPrankId] = useState<string | null>(null);
  const [slug, setSlug] = useState<string | null>(initialSlug ?? null);
  const [loadingExisting, setLoadingExisting] = useState(false);
  const [authed, setAuthed] = useState(!requirePassword);
  const [passwordInput, setPasswordInput] = useState("");

  const hasNewValidRows = rows.some((r) => !r.existing && r.triggerText.trim() && r.file);

  const handleAddRow = () => {
    setRows((prev) => [...prev, { id: makeId(), triggerText: "", file: null }]);
  };

  const handleRemoveRow = (id: string) => {
    setRows((prev) => prev.filter((r) => r.id !== id || r.existing));
  };

  const handleTextChange = (id: string, value: string) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, triggerText: value } : r)));
  };

  const handleFileChange = (id: string, file: File | null) => {
    setRows((prev) =>
      prev.map((r) => {
        if (r.id !== id) return r;
        if (r.previewUrl) {
          URL.revokeObjectURL(r.previewUrl);
        }
        return {
          ...r,
          file,
          previewUrl: file ? URL.createObjectURL(file) : undefined,
        };
      }),
    );
  };

  useEffect(
    () => () => {
      rows.forEach((r) => {
        if (r.previewUrl) URL.revokeObjectURL(r.previewUrl);
      });
    },
    [rows],
  );

  useEffect(() => {
    const loadExisting = async () => {
      if (!slug) return;
      if (requirePassword && !authed) return;
      setLoadingExisting(true);
      setError(null);
      try {
        const data = await fetchPrank(slug);
        setPrankId(data.prank_id);
        setTitle(data.title ?? "");
        if (data.triggers.length > 0) {
          const existingRows: TriggerRow[] = data.triggers.map((t) => ({
            id: t.id,
            triggerText: t.trigger_text,
            file: null,
            previewUrl: `data:image/png;base64,${t.image_base64}`,
            existing: true,
          }));
          setRows([...existingRows, { id: makeId(), triggerText: "", file: null }]);
          setShareUrl(`${window.location.origin}/p/${data.slug}`);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load prank");
      } finally {
        setLoadingExisting(false);
      }
    };
    void loadExisting();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug, authed, requirePassword]);

  const handleCreate = async () => {
    setError(null);
    setShareUrl(null);

    const newRows = rows.filter((r) => !r.existing);
    const invalid = newRows.filter((r) => !r.triggerText.trim() || !r.file);
    if (newRows.length === 0) {
      setError("Add at least one new trigger prompt with a picture.");
      return;
    }
    if (invalid.length > 0) {
      setError("Each new row needs both a prompt and an image.");
      return;
    }

    setIsSaving(true);
    try {
      let effectivePrankId = prankId;
      let effectiveSlug = slug;

      if (!effectivePrankId) {
        const prank = await createPrank(title.trim() || undefined, slug || undefined);
        effectivePrankId = prank.prank_id;
        effectiveSlug = prank.slug;
        setPrankId(prank.prank_id);
        setSlug(prank.slug);
      }

      for (const row of newRows) {
        await addPrankTrigger(effectivePrankId!, row.triggerText.trim(), row.file!);
      }

      // Merge new rows as existing with previews
      const updatedRows: TriggerRow[] = rows.map((r) => (r.existing ? r : null)).filter(Boolean) as TriggerRow[];
      for (const row of newRows) {
        updatedRows.push({
          id: row.id,
          triggerText: row.triggerText.trim(),
          file: null,
          previewUrl: row.previewUrl || (row.file ? URL.createObjectURL(row.file) : undefined),
          existing: true,
        });
      }
      updatedRows.push({ id: makeId(), triggerText: "", file: null });
      setRows(updatedRows);

      const url = `${window.location.origin}/p/${effectiveSlug}`;
      setShareUrl(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create prank");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="page">
      <div className="header" style={{ flexDirection: "column", alignItems: "center" }}>
        <div className="title">Prompt Pics — Prank Builder</div>
        <p style={{ color: "#9ca3af", marginTop: "0.2rem", textAlign: "center" }}>
          Add trigger prompts and prank images, then share the secret link.
          {slug && (
            <span style={{ marginLeft: "0.5rem", color: "#c084fc" }}>
              Editing slug: <strong>{slug}</strong>
            </span>
          )}
        </p>
      </div>

      {requirePassword && !authed ? (
        <div className="panel">
          <div className="section-title">Admin access</div>
          <p style={{ color: "#9ca3af" }}>Enter password to edit this prank.</p>
          <div className="field" style={{ marginTop: "0.75rem" }}>
            <label>Password</label>
            <input
              type="password"
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              placeholder="Password"
            />
          </div>
          <div className="actions-row" style={{ marginTop: "1rem" }}>
            <div className="action-left" />
            <div className="action-right">
              <button
                className="button"
                type="button"
                onClick={() => {
                  if (passwordInput === adminPassword) {
                    setAuthed(true);
                    setError(null);
                  } else {
                    setError("Wrong password.");
                  }
                }}
              >
                Unlock
              </button>
              {error && <span className="error">{error}</span>}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="panel">
            <div className="section-title">Prank settings</div>
            <div className="field">
              <label htmlFor="title">Title (optional)</label>
              <input
                id="title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="My prank link"
              />
            </div>

            <div className="section-title" style={{ marginTop: "1rem" }}>
              Triggers (prompt + image)
            </div>
            <p style={{ color: "#9ca3af", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
              Each new row needs a prompt and a picture. Missing either will block saving.
            </p>

            <div className="trigger-table">
              <div className="trigger-table-head">
                <span>#</span>
                <span>Prompt</span>
                <span>Image</span>
                <span>Actions</span>
              </div>
              {rows.map((row, idx) => (
                <div className="trigger-table-row" key={row.id}>
                  <div className="trigger-col index-col">{idx + 1}</div>
                  <div className="trigger-col prompt-col">
                    <textarea
                      value={row.triggerText}
                      onChange={(e) => handleTextChange(row.id, e.target.value)}
                      placeholder='e.g., "who is the most beautiful person in the world?"'
                      disabled={row.existing}
                    />
                  </div>
                  <div className="trigger-col image-col">
                    {row.existing ? (
                      row.previewUrl && (
                        <div className="file-preview small">
                          <img src={row.previewUrl} alt="Preview" />
                        </div>
                      )
                    ) : (
                      <>
                        <label className="upload-cta inline">
                          <span>{row.file ? "Change image" : "Choose image"}</span>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileChange(row.id, e.target.files?.[0] ?? null)}
                          />
                        </label>
                        {row.previewUrl && (
                          <div className="file-preview small">
                            <img src={row.previewUrl} alt="Preview" />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  <div className="trigger-col action-col">
                    {!row.existing && (
                      <button
                        type="button"
                        className="button"
                        style={{ background: "rgba(248,113,113,0.15)", color: "#fecaca" }}
                        onClick={() => handleRemoveRow(row.id)}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <button type="button" className="button" style={{ marginTop: "1rem" }} onClick={handleAddRow}>
              + Add another
            </button>

            <div className="actions-row" style={{ marginTop: "1.5rem" }}>
              <div className="action-left" />
              <div className="action-right">
                <button
                  className="button"
                  type="button"
                  disabled={isSaving || !hasNewValidRows}
                  onClick={handleCreate}
                >
                  {isSaving ? "Saving..." : "Save prank link"}
                </button>
                {error && <span className="error">{error}</span>}
              </div>
            </div>

            {shareUrl && (
              <div className="glass" style={{ marginTop: "1.5rem" }}>
                <div className="section-title">Share this</div>
                <input
                  readOnly
                  value={shareUrl}
                  onFocus={(e) => e.target.select()}
                  style={{ width: "100%", minWidth: "480px" }}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
