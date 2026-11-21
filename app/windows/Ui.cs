using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Windows.Forms;

namespace GPUEnsembleApp
{
    /// <summary>
    /// The look of the window: the colors of <see cref="Palette"/> (a warm charcoal ground with smoky gold buttons by
    /// default), one color per GPU (smoky gold, purple, blue, ...), joined by a gradient line under the title bar.
    /// </summary>
    static class Ui
    {
        /// <summary>The colors in use; chosen once at start (<see cref="UseTheme"/>), before the window is built.</summary>
        public static Palette Theme { get; private set; } = Palette.Default;

        /// <summary>Uses the named palette (<see cref="Palette.Names"/>); an unknown or empty name keeps the default.</summary>
        public static void UseTheme(string? name) => Theme = Palette.Named(name) ?? Palette.Default;

        public static Color Ground => Theme.Ground;
        public static Color Surface => Theme.Surface;
        public static Color Soft => Theme.Soft;
        public static Color Hover => Theme.Hover;
        public static Color Line => Theme.Line;
        public static Color Line2 => Theme.Line2;
        public static Color Ink => Theme.Ink;
        public static Color Muted => Theme.Muted;
        public static Color DeepBlue => Theme.DeepBlue;
        public static Color Purple => Theme.Purple;
        public static Color Gold => Theme.Gold;
        public static Color Good => Theme.Good;
        public static Color Bad => Theme.Bad;
        public static Color Idle => Theme.Idle;
        public static Color Notice => Theme.Notice;
        public static Color NoticeLine => Theme.NoticeLine;

        /// <summary>One color per GPU (or per PC when there are many), in the order of the model's split.</summary>
        public static Color GpuColor(int i) { var c = Theme.Gpus; return c[((i % c.Length) + c.Length) % c.Length]; }

        public static Color Hex(string s) => Color.FromArgb(Convert.ToInt32(s.Substring(1, 2), 16), Convert.ToInt32(s.Substring(3, 2), 16), Convert.ToInt32(s.Substring(5, 2), 16));

        public static Color Mix(Color a, Color b, double t) => Color.FromArgb(
            (int)Math.Round(a.R + (b.R - a.R) * t), (int)Math.Round(a.G + (b.G - a.G) * t), (int)Math.Round(a.B + (b.B - a.B) * t));

        /// <summary>Whether the window speaks Japanese: the user's choice (<see cref="SetLanguage"/>), else the language of Windows.</summary>
        public static bool Japanese { get; private set; } = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "ja";

        /// <summary>The Japanese or the English text, by <see cref="Japanese"/>.</summary>
        public static string T(string ja, string en) => Japanese ? ja : en;

        static readonly string numbers = Pick(new[] { "Segoe UI Semibold", "Segoe UI" });

        static string Pick(string[] names)
        {
            foreach (var n in names)
            {
                using (var f = new Font(n, 10f))
                    if (string.Equals(f.Name, n, StringComparison.OrdinalIgnoreCase)) return n;
            }
            return SystemFonts.MessageBoxFont.Name;
        }

        public static Font Body { get; private set; } = null!;
        public static Font Small { get; private set; } = null!;
        public static Font Bold { get; private set; } = null!;
        public static Font H1 { get; private set; } = null!;
        public static Font H2 { get; private set; } = null!;

        static Ui() { MakeFonts(); }

        /// <summary>Japanese text reads best in a Japanese face; English keeps Segoe UI.</summary>
        static void MakeFonts()
        {
            var family = Pick(Japanese ? new[] { "Yu Gothic UI", "Meiryo UI", "Segoe UI" } : new[] { "Segoe UI" });
            // The fonts of the language before stay with the controls that still use them.
            Body = new Font(family, 10f);
            Small = new Font(family, 9f);
            Bold = new Font(family, 10f, FontStyle.Bold);
            H1 = new Font(family, 20f, FontStyle.Bold);
            H2 = new Font(family, 13f, FontStyle.Bold);
        }

        /// <summary>Speaks Japanese (true) or English (false) from now on; controls made after this use it.</summary>
        public static void SetLanguage(bool japanese)
        {
            if (japanese == Japanese) return;
            Japanese = japanese;
            MakeFonts();
        }

        public static readonly Font Big = new Font(numbers, 20f);
        public static readonly Font Huge = new Font(numbers, 30f);
        public static readonly Font Brand = new Font(numbers, 14f);
        public static readonly Font Mono = new Font("Consolas", 9.5f);

        static float scale = 1f;

        public static void Init(Control c)
        {
            using (var g = c.CreateGraphics()) scale = g.DpiX / 96f;
        }

        /// <summary>A size in pixels at 100 % display scale, at this display's scale.</summary>
        public static int Px(int px) => (int)Math.Round(px * scale);

        public static GraphicsPath Round(RectangleF r, float radius)
        {
            var p = new GraphicsPath();
            var d = Math.Min(radius * 2, Math.Min(r.Width, r.Height));
            if (d <= 0.5f) { p.AddRectangle(r); return p; }
            p.AddArc(r.X, r.Y, d, d, 180, 90);
            p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
            p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
            p.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
            p.CloseFigure();
            return p;
        }

        public static Label Label(string text, Font? font = null, Color? color = null)
        {
            return new Label
            {
                Text = text,
                Font = font ?? Body,
                ForeColor = color ?? Ink,
                BackColor = Color.Transparent,
                AutoSize = true,
                Margin = new Padding(0, 0, 0, Px(4)),
                UseMnemonic = false,
            };
        }

        /// <summary>A label that wraps at a width.</summary>
        public static Label Wrap(string text, int width, Font? font = null, Color? color = null)
        {
            var l = Label(text, font, color);
            l.MaximumSize = new Size(Math.Max(Px(120), width), 0);
            return l;
        }

        public static Pill Button(string text, bool primary, EventHandler onClick)
        {
            var b = new Pill(primary) { Text = text };
            b.Click += onClick;
            return b;
        }

        public static Card Card(Color? back = null, Color? border = null, Color? band = null) => new Card(back ?? Surface, border ?? Line, band);

        public static FlowLayoutPanel Row(params Control[] controls)
        {
            var p = new FlowLayoutPanel
            {
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = true,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                Margin = new Padding(0, Px(8), 0, 0),
                BackColor = Color.Transparent,
            };
            p.Controls.AddRange(controls);
            return p;
        }

        public static FlowLayoutPanel Column(int bottom = 4)
        {
            return new FlowLayoutPanel
            {
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                Margin = new Padding(0, 0, 0, Px(bottom)),
                BackColor = Color.Transparent,
            };
        }

        /// <summary>A colored dot before a line of text.</summary>
        public static Control Status(string text, Color color)
        {
            var dot = new Dot(color) { Margin = new Padding(0, Px(6), Px(8), 0) };
            var row = Row(dot, Label(text, Bold, color));
            row.WrapContents = false;
            row.Margin = new Padding(0, 0, 0, Px(4));
            return row;
        }

        /// <summary>The color a control stands on: the nearest card, or the nearest parent with a solid color.</summary>
        public static Color Behind(Control c)
        {
            for (var p = c.Parent; p != null; p = p.Parent)
            {
                if (p is Card card) return card.Back;
                if (p.BackColor.A == 255) return p.BackColor;
            }
            return Ground;
        }

        public static string Gib(double gib) => gib >= 10 ? $"{gib:0} GB" : $"{gib:0.#} GB";

        /// <summary>A short name for a card: "RTX 2080 Ti" for "NVIDIA GeForce RTX 2080 Ti 11GB",
        /// "Radeon RX 7900 XTX" for "Vulkan0 AMD Radeon RX 7900 XTX", "Arc B580" for "Vulkan1 Intel(R) Arc(TM) B580 Graphics".</summary>
        public static string ShortGpu(string name)
        {
            var n = name.Replace("NVIDIA ", "").Replace("GeForce ", "").Replace("AMD ", "").Replace("Intel(R) ", "")
                        .Replace("(TM)", "").Replace(" Graphics", "");
            if (n.StartsWith("Vulkan", StringComparison.Ordinal) && n.IndexOf(' ') > 6 && char.IsDigit(n[6])) n = n.Substring(n.IndexOf(' ') + 1);
            var on = n.IndexOf(" on ", StringComparison.Ordinal);
            if (on > 0) n = n.Substring(0, on);
            if (n.StartsWith("GPU", StringComparison.Ordinal) && n.IndexOf(' ') > 0 && char.IsDigit(n[3])) n = n.Substring(n.IndexOf(' ') + 1);
            var parts = n.Split(' ');
            if (parts.Length > 1 && parts[parts.Length - 1].EndsWith("GB", StringComparison.Ordinal)) n = string.Join(" ", parts, 0, parts.Length - 1);
            return n;
        }
    }

    /// <summary>A white box with rounded corners and a thin border; with a band, a colored strip on top.</summary>
    sealed class Card : FlowLayoutPanel
    {
        readonly Color back, border;
        readonly Color? band;

        public Color Back => back;

        public Card(Color back, Color border, Color? band)
        {
            this.back = back; this.border = border; this.band = band;
            FlowDirection = FlowDirection.TopDown;
            WrapContents = false;
            AutoSize = true;
            AutoSizeMode = AutoSizeMode.GrowAndShrink;
            BackColor = Color.Transparent;
            Padding = new Padding(Ui.Px(22), Ui.Px(band.HasValue ? 22 : 18), Ui.Px(22), Ui.Px(18));
            Margin = new Padding(0, 0, Ui.Px(14), Ui.Px(14));
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw | ControlStyles.SupportsTransparentBackColor, true);
        }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Ui.Behind(this));
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var r = new RectangleF(0.5f, 0.5f, Width - 1.5f, Height - 1.5f);
            using (var path = Ui.Round(r, Ui.Px(14)))
            {
                using (var b = new SolidBrush(back)) g.FillPath(b, path);
                if (band.HasValue)
                {
                    var clip = g.Clip;
                    g.SetClip(path);
                    using (var b = new SolidBrush(band.Value)) g.FillRectangle(b, 0, 0, Width, Ui.Px(6));
                    g.Clip = clip;
                }
                using (var p = new Pen(border)) g.DrawPath(p, path);
            }
        }
    }

    /// <summary>
    /// A rounded button: deep blue with light gold text, or white with a thin border. A plain control drawn
    /// whole by itself: a WinForms Button kept drawing parts of its own (a square box with bits of the text
    /// around it) behind the rounded shape.
    /// </summary>
    sealed class Pill : Control
    {
        readonly bool primary;
        bool hover, down;

        public Pill(bool primary)
        {
            this.primary = primary;
            Font = primary ? Ui.Bold : Ui.Body;
            Height = Ui.Px(44);
            Margin = new Padding(0, 0, Ui.Px(10), 0);
            Cursor = Cursors.Hand;
            TabStop = true;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint |
                     ControlStyles.ResizeRedraw | ControlStyles.Selectable | ControlStyles.StandardClick, true);
            SetStyle(ControlStyles.StandardDoubleClick, false);
        }

        protected override void OnTextChanged(EventArgs e)
        {
            base.OnTextChanged(e);
            Width = TextRenderer.MeasureText(Text, Font).Width + Ui.Px(primary ? 56 : 44);
            Invalidate();
        }

        protected override void OnEnabledChanged(EventArgs e)
        {
            base.OnEnabledChanged(e);
            Cursor = Enabled ? Cursors.Hand : Cursors.Default;
            Invalidate();
        }

        protected override void OnMouseEnter(EventArgs e) { hover = true; Invalidate(); base.OnMouseEnter(e); }
        protected override void OnMouseLeave(EventArgs e) { hover = false; down = false; Invalidate(); base.OnMouseLeave(e); }
        protected override void OnMouseDown(MouseEventArgs e) { if (e.Button == MouseButtons.Left) { down = true; Focus(); Invalidate(); } base.OnMouseDown(e); }
        protected override void OnMouseUp(MouseEventArgs e) { down = false; Invalidate(); base.OnMouseUp(e); }
        protected override void OnGotFocus(EventArgs e) { Invalidate(); base.OnGotFocus(e); }
        protected override void OnLostFocus(EventArgs e) { Invalidate(); base.OnLostFocus(e); }

        protected override bool IsInputKey(Keys keyData) => keyData == Keys.Enter || keyData == Keys.Space || base.IsInputKey(keyData);

        protected override void OnKeyUp(KeyEventArgs e)
        {
            base.OnKeyUp(e);
            if (Enabled && (e.KeyCode == Keys.Enter || e.KeyCode == Keys.Space)) OnClick(EventArgs.Empty);
        }

        /// <summary>The corners show the color of what the button stands on (a card, or the page).</summary>
        protected override void OnPaintBackground(PaintEventArgs e) => e.Graphics.Clear(Ui.Behind(this));

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var r = new RectangleF(0.5f, 0.5f, Width - 1.5f, Height - 1.5f);
            Color fill, text, edge;
            if (!Enabled) { fill = Ui.Soft; text = Ui.Idle; edge = Ui.Line; }
            else if (primary) { fill = hover || down ? Ui.Theme.ButtonHover : Ui.Theme.Button; text = Ui.Theme.ButtonText; edge = fill; }
            else { fill = hover || down ? Ui.Soft : Ui.Surface; text = Ui.Ink; edge = Ui.Line2; }
            using (var path = Ui.Round(r, Height / 2f))
            {
                using (var b = new SolidBrush(fill)) g.FillPath(b, path);
                using (var p = new Pen(edge)) g.DrawPath(p, path);
            }
            TextRenderer.DrawText(g, Text, Font, ClientRectangle, text, fill,
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine);
            if (Focused && ShowFocusCues)
                using (var p = new Pen(Ui.Gold, Ui.Px(2)))
                using (var path = Ui.Round(new RectangleF(Ui.Px(2), Ui.Px(2), Width - Ui.Px(5), Height - Ui.Px(5)), (Height - Ui.Px(4)) / 2f))
                    g.DrawPath(p, path);
        }
    }

    sealed class Dot : Control
    {
        readonly Color color;
        public Dot(Color color)
        {
            this.color = color;
            Size = new Size(Ui.Px(10), Ui.Px(10));
            SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);
            BackColor = Color.Transparent;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using (var b = new SolidBrush(color)) e.Graphics.FillEllipse(b, 0, 0, Width - 1, Height - 1);
        }
    }

    /// <summary>A rounded bar filled to a fraction, in one color or a gradient of two.</summary>
    sealed class Meter : Control
    {
        double fraction;
        readonly Color from, to;

        /// <summary>How full the bar is, 0 to 1; setting it redraws only the bar.</summary>
        public double Fraction
        {
            get => fraction;
            set
            {
                var v = Math.Max(0, Math.Min(1, value));
                if (Math.Abs(v - fraction) < 0.0005) return;
                fraction = v;
                Invalidate();
            }
        }

        public Meter(double fraction, Color from, Color to, int width, int height = 8)
        {
            this.fraction = Math.Max(0, Math.Min(1, fraction));
            this.from = from; this.to = to;
            Size = new Size(width, Ui.Px(height));
            Margin = new Padding(0, Ui.Px(4), 0, Ui.Px(6));
            SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.ResizeRedraw, true);
            BackColor = Color.Transparent;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var r = new RectangleF(0, 0, Width - 1, Height - 1);
            using (var track = Ui.Round(r, Height / 2f))
            using (var b = new SolidBrush(Ui.Soft)) g.FillPath(b, track);
            var w = (float)((Width - 1) * fraction);
            if (w < 1) return;
            using (var fill = Ui.Round(new RectangleF(0, 0, Math.Max(w, Height), Height - 1), Height / 2f))
            using (var b = new LinearGradientBrush(new RectangleF(0, 0, Math.Max(w, 2), Height), from, to, 0f))
                g.FillPath(b, fill);
        }
    }

    /// <summary>One PC's row in the list of PCs: a color strip on the left, a line under it; click to open.</summary>
    sealed class PcRow : Panel
    {
        readonly Color color;
        readonly bool open;

        public PcRow(Color color, bool open)
        {
            this.color = color; this.open = open;
            Margin = new Padding(0);
            Cursor = Cursors.Hand;
            BackColor = open ? Ui.Theme.OpenRow : Ui.Surface;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
        }

        public void PerformClick() => OnClick(EventArgs.Empty);

        protected override void OnPaintBackground(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(BackColor);
            using (var b = new SolidBrush(color)) g.FillRectangle(b, 0, Ui.Px(10), Ui.Px(5), Height - Ui.Px(20));
            using (var p = new Pen(Ui.Theme.RowLine)) g.DrawLine(p, 0, Height - 1, Width, Height - 1);
        }
    }

    /// <summary>
    /// The memory of a PC's GPUs side by side: one short bar per GPU, as wide as its memory, filled to what
    /// the model uses on it.
    /// </summary>
    sealed class MultiMeter : Control
    {
        readonly (double capacity, double used)[] gpus;
        readonly Color color;

        public MultiMeter((double capacity, double used)[] gpus, Color color, int width)
        {
            this.gpus = gpus; this.color = color;
            Size = new Size(width, Ui.Px(8));
            Margin = new Padding(0, Ui.Px(6), 0, Ui.Px(4));
            SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.ResizeRedraw, true);
            BackColor = Color.Transparent;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            double total = 0;
            foreach (var x in gpus) total += x.capacity;
            if (total <= 0) return;
            float gap = gpus.Length > 1 ? Ui.Px(3) : 0, room = Width - gap * (gpus.Length - 1), at = 0;
            using (var track = new SolidBrush(Ui.Soft))
            using (var fill = new SolidBrush(color))
                foreach (var x in gpus)
                {
                    var w = (float)(room * x.capacity / total);
                    g.FillRectangle(track, at, 0, w, Height);
                    if (x.capacity > 0) g.FillRectangle(fill, at, 0, (float)(w * Math.Min(1, x.used / x.capacity)), Height);
                    at += w + gap;
                }
        }
    }

    /// <summary>
    /// How a model's layers are split over the GPUs: one stretch per GPU, in its color, each blending into the
    /// next at the border, as in the design.
    /// </summary>
    sealed class SplitBar : Control
    {
        readonly (int layers, Color color)[] parts;

        public SplitBar((int layers, Color color)[] parts, int width)
        {
            this.parts = parts;
            Size = new Size(width, Ui.Px(14));
            Margin = new Padding(0, Ui.Px(6), 0, Ui.Px(8));
            SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.ResizeRedraw, true);
            BackColor = Color.Transparent;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var total = 0;
            foreach (var p in parts) total += p.layers;
            if (total == 0) return;
            using (var outline = Ui.Round(new RectangleF(0, 0, Width - 1, Height - 1), Height / 2f))
            {
                g.SetClip(outline);
                float x = 0;
                for (int i = 0; i < parts.Length; i++)
                {
                    var w = (Width - 1) * parts[i].layers / (float)total;
                    if (w <= 0) continue;
                    var next = i + 1 < parts.Length ? parts[i + 1].color : parts[i].color;
                    if (i + 1 < parts.Length && next == parts[i].color)
                    {
                        // The next GPU is on the same PC (one color per PC): a thin light line marks the border.
                        using (var b = new SolidBrush(parts[i].color)) g.FillRectangle(b, x, 0, w + 1, Height);
                        using (var pen = new Pen(Color.FromArgb(150, Ui.Theme.Tick))) g.DrawLine(pen, x + w, Ui.Px(3), x + w, Height - Ui.Px(3));
                        x += w;
                        continue;
                    }
                    // A short blend at the border only, so a GPU with few layers still shows its own color.
                    var edge = Math.Min(0.22f, Ui.Px(10) / w);
                    var blend = new ColorBlend
                    {
                        Colors = new[] { parts[i].color, parts[i].color, next },
                        Positions = new[] { 0f, 1f - edge, 1f },
                    };
                    using (var b = new LinearGradientBrush(new RectangleF(x - 0.5f, 0, w + 1, Height), parts[i].color, next, 0f) { InterpolationColors = blend })
                        g.FillRectangle(b, x, 0, w + 1, Height);
                    x += w;
                }
                g.ResetClip();
            }
        }
    }

    /// <summary>Three overlapping circles in the GPU colors: the mark of the app.</summary>
    sealed class Logo : Control
    {
        public Logo()
        {
            Size = new Size(Ui.Px(34), Ui.Px(22));
            Margin = new Padding(0, Ui.Px(2), Ui.Px(10), 0);
            SetStyle(ControlStyles.SupportsTransparentBackColor | ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);
            BackColor = Color.Transparent;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            float d = Height - 2, step = (Width - d) / 2f;
            var colors = new[] { Ui.GpuColor(0), Ui.GpuColor(1), Ui.GpuColor(2) };
            for (int i = 0; i < 3; i++)
                using (var b = new SolidBrush(Color.FromArgb(i == 0 ? 220 : 190, colors[i])))
                    g.FillEllipse(b, i * step, 1, d, d);
        }
    }

    /// <summary>The thin line under the title bar: deep blue to purple to smoky gold.</summary>
    sealed class GradientLine : Control
    {
        public GradientLine()
        {
            Dock = DockStyle.Top;
            Height = Ui.Px(4);
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.ResizeRedraw, true);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            if (Width < 2) return;
            using (var b = new LinearGradientBrush(ClientRectangle, Ui.GpuColor(2), Ui.GpuColor(0), 0f)
            {
                InterpolationColors = new ColorBlend { Colors = new[] { Ui.GpuColor(2), Ui.GpuColor(1), Ui.GpuColor(0) }, Positions = new[] { 0f, 0.5f, 1f } },
            })
                e.Graphics.FillRectangle(b, ClientRectangle);
        }
    }
}
