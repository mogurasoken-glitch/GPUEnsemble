using System;
using System.Drawing;

namespace GPUEnsembleApp
{
    /// <summary>
    /// The window's colors. "charcoal" (the default since 2026-11-19) is a warm near-black with smoky gold
    /// buttons; "light" is the earlier near-white look. <c>--theme NAME</c> or <c>theme</c> in app-settings.json picks one.
    /// </summary>
    sealed class Palette
    {
        public string Name = "";
        /// <summary>Dark ground: Windows draws the title bar dark too.</summary>
        public bool Dark;
        public Color Ground, Chrome, Surface, Soft, Hover, Line, Line2, Ink, Muted;
        /// <summary>Accent colors for words (statuses, tags); on a dark ground lighter than the GPU colors.</summary>
        public Color DeepBlue, Purple, Gold;
        public Color Good, Bad, Idle, Notice, NoticeLine;
        public Color Button, ButtonHover, ButtonText;
        /// <summary>An open row in a list, the line under each row, and the thin border between two GPUs of one PC.</summary>
        public Color OpenRow, RowLine, Tick;
        /// <summary>One color per GPU (or per PC), in the order of the model's split.</summary>
        public Color[] Gpus = Array.Empty<Color>();

        // Not Ui.Hex: Ui's own start reads Palette.Default, so Palette must not need Ui to start.
        static Color Hex(string s) => Color.FromArgb(Convert.ToInt32(s.Substring(1, 2), 16), Convert.ToInt32(s.Substring(3, 2), 16),
                                                     Convert.ToInt32(s.Substring(5, 2), 16));

        public static readonly Palette Charcoal = new Palette
        {
            Name = "charcoal",
            Dark = true,
            Ground = Hex("#201E24"),
            Chrome = Hex("#18171B"),
            Surface = Hex("#2B2830"),
            Soft = Hex("#36323C"),
            Hover = Hex("#25222A"),
            Line = Hex("#3B3742"),
            Line2 = Hex("#514B58"),
            Ink = Hex("#F2EDE6"),
            Muted = Hex("#B5AEA6"),
            DeepBlue = Hex("#93A5F2"),
            Purple = Hex("#BC9BE6"),
            Gold = Hex("#DDBA6C"),
            Good = Hex("#56C996"),
            Bad = Hex("#F08A7E"),
            Idle = Hex("#8F8880"),
            Notice = Hex("#373024"),
            NoticeLine = Hex("#7A6537"),
            Button = Hex("#D2AA55"),
            ButtonHover = Hex("#E0BD6E"),
            ButtonText = Hex("#18171B"),
            OpenRow = Hex("#322E37"),
            RowLine = Hex("#3B3742"),
            Tick = Hex("#18171B"),
            Gpus = new[] { Hex("#D2AA55"), Hex("#A47AD6"), Hex("#7088EA"), Hex("#4FA8B8"), Hex("#CF7393"), Hex("#8DAA55"),
                           Hex("#E08E55"), Hex("#5FA0E6"), Hex("#C48BE0"), Hex("#B5AE5A"), Hex("#D88A8A"), Hex("#6FB59A") },
        };

        public static readonly Palette Light = new Palette
        {
            Name = "light",
            Dark = false,
            Ground = Hex("#F7F7FB"),
            Chrome = Color.White,
            Surface = Color.White,
            Soft = Hex("#EEEFF7"),
            Hover = Hex("#F6F6FB"),
            Line = Hex("#E2E3EE"),
            Line2 = Hex("#C8CAE0"),
            Ink = Hex("#121433"),
            Muted = Hex("#565A7A"),
            DeepBlue = Hex("#1E2A78"),
            Purple = Hex("#5B2A86"),
            Gold = Hex("#B08D3C"),
            Good = Hex("#0F7B4F"),
            Bad = Hex("#B3261E"),
            Idle = Hex("#8A8DA6"),
            Notice = Hex("#FBF5E6"),
            NoticeLine = Hex("#E2CC93"),
            Button = Hex("#1E2A78"),
            ButtonHover = Hex("#5B2A86"),
            ButtonText = Hex("#F3E2B0"),
            OpenRow = Hex("#FAFAFD"),
            RowLine = Hex("#EEEFF6"),
            Tick = Color.White,
            Gpus = new[] { Hex("#B08D3C"), Hex("#5B2A86"), Hex("#1E2A78"), Hex("#2E7D8C"), Hex("#8C3B5E"), Hex("#4F6B2E"),
                           Hex("#C2703D"), Hex("#3F5FB0"), Hex("#7A4FB5"), Hex("#6B6630"), Hex("#A0525E"), Hex("#2F4A6B") },
        };

        public static Palette Default => Charcoal;

        public static readonly string[] Names = { "charcoal", "light" };

        public static Palette? Named(string? name)
        {
            switch ((name ?? "").Trim().ToLowerInvariant())
            {
                case "charcoal": return Charcoal;
                case "light": return Light;
                default: return null;
            }
        }
    }
}
