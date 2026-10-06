"""The hosting files (web demo item 17): the key never goes into the image or the repo."""

from webdemo.settings import ROOT


def test_the_key_stays_out_of_the_image():
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in ignore and "logs" in ignore and ".cache" in ignore
    docker = (ROOT / "Dockerfile").read_text()
    assert "NEBIUS_API_KEY" not in docker and ".env" not in docker
    compose = (ROOT / "deploy" / "compose.yaml").read_text()
    assert "/etc/deixis/webdemo.env" in compose
    # only the placeholder in a comment, never a value
    assert compose.count("NEBIUS_API_KEY=") == compose.count("NEBIUS_API_KEY=...")


def test_https_in_front_and_one_proxy_hop():
    caddy = (ROOT / "deploy" / "Caddyfile").read_text()
    assert "reverse_proxy webdemo:8000" in caddy and "noindex" in caddy
    assert "PROXY_HOPS=1" in (ROOT / "Dockerfile").read_text()
    run = (ROOT / "deploy" / "run.sh").read_text()
    assert "*deixis*|*rahul*" in run  # the host name must not name the project or rahul
