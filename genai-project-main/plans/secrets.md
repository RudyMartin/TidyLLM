Resolving Netlify Secret‑Scanning Failures for Supabase Anon Key
When you deploy a Vite/Supabase project to Netlify, you may encounter an error like this:

swift
Copy
Edit
Secrets scanning found secrets in files during build.
Secret env var "VITE_SUPABASE_ANON_KEY"'s value detected: found value in `dist/assets/index-bad8aaa4.js`.
This happens because Vite embeds import.meta.env variables into the compiled JavaScript bundle. Your Supabase client is created like this:

ts
Copy
Edit
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
At build time, VITE_SUPABASE_ANON_KEY is replaced with its value, and the secret scanner flags it. Although Supabase’s “anon key” is publishable, Netlify’s smart detection sees any key‑shaped string as sensitive and fails the build.

Recommended fix
Netlify provides configuration variables to control secret scanning. You can instruct Netlify to omit scanning for specific keys using the SECRETS_SCAN_OMIT_KEYS environment variable. The documentation notes that you can supply a comma‑separated list of environment variable names that should not be scanned
docs.netlify.com
.

To fix the build:

Open your site on Netlify and go to Site settings → Environment variables.

Add a new variable:

Key: SECRETS_SCAN_OMIT_KEYS

Value: VITE_SUPABASE_ANON_KEY

This tells Netlify that the value of VITE_SUPABASE_ANON_KEY may legitimately appear in your build and should not trigger a failure. You can include multiple keys separated by commas if you have other false positives.

Redeploy your site. Netlify should no longer fail the build due to the Supabase key.

Alternative options (less recommended)
Disable secret scanning entirely by adding SECRETS_SCAN_ENABLED = false. The docs explain that setting this variable to false disables all secret‑scanning protections
docs.netlify.com
. This is only advisable if you have another process protecting secrets.

Use SECRETS_SCAN_SMART_DETECTION_OMIT_VALUES with the actual string value of the key. This safelists specific values rather than variable names, but it can become hard to maintain as keys change.

Refactor the application so the Supabase client is initialized on the server side (e.g. in a Netlify function) instead of embedding the key into client JavaScript. This approach is more secure because the key never appears in the bundle, but it requires reworking your API calls.

Summary
Netlify’s secret scanner blocks deployments when it detects secret values in the build output. Because Vite inlines the Supabase ANON key in client JavaScript, you should instruct Netlify to ignore that key by setting SECRETS_SCAN_OMIT_KEYS=VITE_SUPABASE_ANON_KEY. The configuration variables for secret scanning—SECRETS_SCAN_ENABLED, SECRETS_SCAN_OMIT_KEYS, and SECRETS_SCAN_OMIT_PATHS—are documented in Netlify’s secret scanning guide
docs.netlify.com
.



It covers why the build fails, how to use SECRETS_SCAN_OMIT_KEYS to safelist the key, and alternative options for handling secret scanning on Netlify, with citations from Netlify’s documentation.
