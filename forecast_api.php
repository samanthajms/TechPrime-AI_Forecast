?php
// Proxy: browser/PHP pages -> Render forecast service. Keeps the API key server-side.
// Usage: forecast_api.php?type=demand|revenue|metrics&horizon=3&category=MEMORY&top=50
// .env (TechPrime-AI/.env, never committed):  FORECAST_URL=https://techprime-forecast.onrender.com   FORECAST_API_KEY=<same value as on Render>
declare(strict_types=1);
header('Content-Type: application/json');
if (session_status() === PHP_SESSION_NONE) { session_start(); }

// Same convention as the other JSON endpoints: check the role yourself and return {ok:false,error} + 403.
if (!in_array($_SESSION['role'] ?? '', ['admin', 'retail_officer'], true)) {
    http_response_code(403);
    echo json_encode(['ok' => false, 'error' => 'forbidden']);
    exit;
}

$base = rtrim((string)($_ENV['FORECAST_URL'] ?? getenv('FORECAST_URL') ?: ''), '/');
$key  = (string)($_ENV['FORECAST_API_KEY'] ?? getenv('FORECAST_API_KEY') ?: '');
if ($base === '' || $key === '') {
    http_response_code(500);
    echo json_encode(['ok' => false, 'error' => 'Forecast service is not configured']);
    exit;
}

$routes  = ['demand' => '/api/forecast/demand', 'revenue' => '/api/forecast/revenue', 'metrics' => '/api/metrics'];
$type    = $_GET['type'] ?? 'demand';
if (!isset($routes[$type])) { http_response_code(400); echo json_encode(['ok' => false, 'error' => 'bad type']); exit; }

$query = ['horizon' => max(1, min((int)($_GET['horizon'] ?? 3), 6)), 'top' => max(1, min((int)($_GET['top'] ?? 50), 300))];
if (!empty($_GET['category'])) { $query['category'] = (string)$_GET['category']; }

$ch = curl_init($base . $routes[$type] . '?' . http_build_query($query));
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_CONNECTTIMEOUT => 10,
    CURLOPT_TIMEOUT        => 75,               // Render free tier can need ~50 s to wake from idle
    CURLOPT_HTTPHEADER     => ['X-API-Key: ' . $key],
]);
$response = curl_exec($ch);
$code     = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
$err      = curl_error($ch);
curl_close($ch);

if ($response === false || $err) {
    http_response_code(503);
    echo json_encode(['ok' => false, 'error' => 'Forecast service unreachable (it may be waking up - retry in a minute)']);
    exit;
}
http_response_code($code ?: 502);
echo $response;
