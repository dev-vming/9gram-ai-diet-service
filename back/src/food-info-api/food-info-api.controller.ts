import { Controller, Get, UseGuards } from "@nestjs/common";
import { FoodInfoApiService } from "./food-info-api.service";
import { ApiTags } from "@nestjs/swagger";
import { isLoggedInGuard } from "src/auth/utils/guards/isLoggedin.guard";

@Controller("food-info-api")
@ApiTags("Food Info (외부 api) API")
export class FoodInfoApiController {
  constructor(private foodInfoApiservice: FoodInfoApiService) {}

  // 전국식품영양(음식) api
  @Get("/portalAPI")
  @UseGuards(isLoggedInGuard)
  getDataPortalFood() {
    return this.foodInfoApiservice.getDataPortal();
  }

  // 전국식품영양(가공식품) api
  @Get("/portal-processAPI")
  @UseGuards(isLoggedInGuard)
  getDataPortalProcessFood() {
    return this.foodInfoApiservice.getDataPortalProcess();
  }
}
