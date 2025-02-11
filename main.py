from abc import ABC

import numpy as np
import pygame as pg
from numba import njit, prange

FPS = 120
G = 10 ** 4

class GameObject(ABC):
    def handle_event(self, event: pg.event.Event) -> bool:
        """Обрабатывает событие Pygame."""
        return False

    def update(self, dt: float):
        """Обновляет состояние объекта."""
        pass

    def draw(self, surface: pg.Surface):
        """Рисует объект на поверхности."""
        pass


class Body:
    def __init__(self, x, y, m, r, vx=0.0, vy=0.0):
        self.pos = np.array([float(x), float(y)], dtype=np.float64)
        self.m = m
        self.r = r
        self.v = np.array([float(vx), float(vy)], dtype=np.float64)

    def update(self, pos, v):
        self.pos = pos
        self.v = v

class StationaryBody(Body):
    def update(self, pos, v):
        pass  # Статичное тело не двигается

@njit(parallel=True)
def compute_forces(positions, masses, radii):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    
    for i in prange(N):
        for j in range(i + 1, N):  # Проход только по уникальным парам
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            r = max(r, radii[i] + radii[j])  # Минимальное расстояние
            
            if r > 0:
                F = (G * masses[i] * masses[j] / r**3) * r_vec
                forces[i] += F
                forces[j] -= F  # Симметрично

    return forces

@njit
def update_positions(positions, velocities, forces, masses, dt):
    for i in range(positions.shape[0]):
        acceleration = forces[i] / masses[i]
        velocities[i] += acceleration * dt
        positions[i] += velocities[i] * dt

class Simulation:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def tick(self, dt):
        N = len(self.objects)
        positions = np.array([body.pos for body in self.objects], dtype=np.float64)
        velocities = np.array([body.v for body in self.objects], dtype=np.float64)
        masses = np.array([body.m for body in self.objects], dtype=np.float64)
        radii = np.array([body.r for body in self.objects], dtype=np.float64)

        forces = compute_forces(positions, masses, radii)
        update_positions(positions, velocities, forces, masses, dt)

        # Записываем новые значения обратно в объекты
        for i in range(N):
            self.objects[i].update(positions[i], velocities[i])


class TransformableSpace(GameObject):
    def __init__(self):
        self.scale = 1
        self.offset_x = 0
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        
        self.transformation_matrix = self.update_transformation_matrix()
        self.simulation = Simulation([StationaryBody(300, 300, 10000, 10)])
    
    def update_transformation_matrix(self):
        """Обновляет матрицу трансформации."""
        return np.array([
            [self.scale, 0, self.offset_x],
            [0, self.scale, self.offset_y],
            [0, 0, 1]
        ])
    
    def world_to_screen(self, pos):
        """Преобразует координаты из виртуального пространства в экранные."""
        virtual_point = np.array([*pos, 1])
        screen_point = self.transformation_matrix @ virtual_point
        return int(screen_point[0]), int(screen_point[1])
    
    def screen_to_world(self, pos):
        """Преобразует экранные координаты обратно в виртуальные."""
        inverse_matrix = np.linalg.inv(self.transformation_matrix)
        screen_point = np.array([*pos, 1])  # Однородные координаты
        virtual_point = inverse_matrix @ screen_point  # Умножение на обратную матрицу
        return virtual_point[:2]  # Возвращаем x, y
    
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 2:  # Средняя кнопка мыши
                self.is_dragging = True
                self.last_mouse_pos = event.pos
                return True
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 2:
                self.is_dragging = False
                return True
        elif event.type == pg.MOUSEMOTION and self.is_dragging:
            dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
            self.offset_x += dx
            self.offset_y += dy
            self.last_mouse_pos = event.pos
            self.transformation_matrix = self.update_transformation_matrix()
        elif event.type == pg.MOUSEWHEEL:
            mouse_x, mouse_y = pg.mouse.get_pos()
            world_x = (mouse_x - self.offset_x) / self.scale
            world_y = (mouse_y - self.offset_y) / -self.scale
            
            self.scale *= 1.1 if event.y > 0 else 0.9
            
            self.offset_x = mouse_x - world_x * self.scale
            self.offset_y = mouse_y - world_y * -self.scale
            self.transformation_matrix = self.update_transformation_matrix()
        return False

    def update(self, dt):
        self.simulation.tick(dt)

    def draw(self, surface):
        for obj in self.simulation.objects:
            pg.draw.circle(surface, pg.color.Color(70, 130, 180), self.world_to_screen(obj.pos), max(3, obj.r * self.scale))

    def add_object(self, pos, m, r, mouse_last=None):
        r_vec = np.array(self.screen_to_world(mouse_last)) - np.array(self.screen_to_world(pos))
        pos = self.screen_to_world(pos)
        body = Body(*pos, m, r, r_vec[0], r_vec[1])
        self.simulation.objects.append(body)

class BodyCreationTool(GameObject):
    def __init__(self, space: TransformableSpace):
        self.clicked = False
        self.click_pos = (0, 0)
        self.space = space

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.clicked = True
                self.click_pos = pg.mouse.get_pos()
                return True
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:
                self.clicked = False

                self.space.add_object(self.click_pos, 10, 5, pg.mouse.get_pos())
        return False

    def draw(self, surface):
        if self.clicked:
            pg.draw.line(surface, "red", self.click_pos, pg.mouse.get_pos())

class Main:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1280, 720))
        self.clock = pg.time.Clock()
        self.running = True
        self.dt = 0.0

        self.space = TransformableSpace()
        self.body_creation_tool = BodyCreationTool(self.space)

    def main_loop(self):
        while self.running:
            self.update_events()
            self.space.update(self.dt)

            self.screen.fill(pg.color.Color(13, 13, 26))
            self.draw()
            pg.display.flip()

            self.dt = self.clock.tick(FPS) / 1000

    def update_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif self.body_creation_tool.handle_event(event):
                pass
            else:
                self.space.handle_event(event)

    def draw(self):
        self.space.draw(self.screen)
        self.body_creation_tool.draw(self.screen)


if __name__ == "__main__":
    game = Main()
    game.main_loop()